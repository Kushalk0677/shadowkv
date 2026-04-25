from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .cache import TieredStateBank


@dataclass
class SpeculationDecision:
    prefix_tokens: Tuple[int, ...]
    score: float
    target_tier: str = 'cpu'
    expected_benefit_ms: float = 0.0
    expected_cost_ms: float = 0.0


class SpeculationPolicy:
    def rank(self, bank: TieredStateBank, budget_k: int, prefer_gpu: bool = False) -> List[SpeculationDecision]:
        raise NotImplementedError


class FrequencyPolicy(SpeculationPolicy):
    def __init__(self, min_frequency: float = 0.20, min_prefix_len: int = 12, max_prefix_len: int = 64, min_observations: int = 3):
        self.min_frequency = min_frequency
        self.min_prefix_len = min_prefix_len
        self.max_prefix_len = max_prefix_len
        self.min_observations = min_observations

    def rank(self, bank: TieredStateBank, budget_k: int, prefer_gpu: bool = False) -> List[SpeculationDecision]:
        tier = 'gpu' if prefer_gpu else 'cpu'
        decisions: List[SpeculationDecision] = []
        candidates = bank.get_candidate_stats(max_prefix_len=self.max_prefix_len, exclude_stored=True)
        for prefix, freq, observations, branching_factor in candidates:
            if len(prefix) < self.min_prefix_len or freq < self.min_frequency or observations < self.min_observations:
                continue
            effective_len = min(len(prefix), self.max_prefix_len)
            confidence = min(observations / 6.0, 1.0)
            branch_bonus = 1.0 + 0.10 * min(max(branching_factor - 1, 0), 4)
            score = float(freq) * confidence * branch_bonus * (1.0 + 0.08 * effective_len)
            decisions.append(
                SpeculationDecision(
                    prefix_tokens=prefix,
                    score=score,
                    target_tier=tier,
                    expected_benefit_ms=freq * effective_len,
                    expected_cost_ms=max(2.0, effective_len * 0.25),
                )
            )
        decisions.sort(key=lambda d: (d.score, len(d.prefix_tokens)), reverse=True)
        return decisions[:budget_k]


class CostAwareSlackPolicy(SpeculationPolicy):
    """Rank prefixes by expected utility during idle time.

    The model is:
        expected_utility = expected_saved_prefill_ms - discounted_speculative_cost_ms

    where expected_saved_prefill_ms is driven by a reuse probability estimate built
    from global frequency, recent support, recent streaks, and observation count.
    """

    def __init__(
        self,
        min_frequency: float = 0.18,
        ms_per_token: float = 1.2,
        fixed_prefill_overhead_ms: float = 5.0,
        memory_penalty_per_mb: float = 0.9,
        kv_mb_per_token: float = 0.0015,
        idle_cost_fraction: float = 0.50,
        gpu_idle_cost_fraction: float = 0.35,
        benefit_cost_ratio: float = 1.05,
        max_admissions_per_idle: int = 1,
        min_prefix_len: int = 12,
        preferred_prefix_len: int = 48,
        max_prefix_len: int = 128,
        min_observations: int = 3,
        min_expected_net_ms: float = 6.0,
        min_recent_support: float = 0.05,
        recent_support_weight: float = 0.55,
        recent_streak_weight: float = 0.20,
        observation_weight: float = 0.20,
        global_frequency_weight: float = 0.25,
        reuse_discount: float = 0.92,
        max_reuse_probability: float = 0.98,
        min_utility_score: float = 0.0,
        reuse_horizon_s: float = 0.75,
        bootstrap_horizon_requests: float = 0.75,
        branching_weight: float = 0.12,
        dominance_tolerance: float = 0.90,
    ):
        self.min_frequency = min_frequency
        self.ms_per_token = ms_per_token
        self.fixed_prefill_overhead_ms = fixed_prefill_overhead_ms
        self.memory_penalty_per_mb = memory_penalty_per_mb
        self.kv_mb_per_token = kv_mb_per_token
        self.idle_cost_fraction = idle_cost_fraction
        self.gpu_idle_cost_fraction = gpu_idle_cost_fraction
        self.benefit_cost_ratio = benefit_cost_ratio
        self.max_admissions_per_idle = max_admissions_per_idle
        self.min_prefix_len = min_prefix_len
        self.preferred_prefix_len = preferred_prefix_len
        self.max_prefix_len = max_prefix_len
        self.min_observations = min_observations
        self.min_expected_net_ms = min_expected_net_ms
        self.min_recent_support = min_recent_support
        self.recent_support_weight = recent_support_weight
        self.recent_streak_weight = recent_streak_weight
        self.observation_weight = observation_weight
        self.global_frequency_weight = global_frequency_weight
        self.reuse_discount = reuse_discount
        self.max_reuse_probability = max_reuse_probability
        self.min_utility_score = min_utility_score
        self.reuse_horizon_s = reuse_horizon_s
        self.bootstrap_horizon_requests = bootstrap_horizon_requests
        self.branching_weight = branching_weight
        self.dominance_tolerance = dominance_tolerance

    def _estimate_memory_mb(self, prefix: Tuple[int, ...]) -> float:
        return max(len(prefix) * self.kv_mb_per_token, 0.002)

    def _length_quality(self, prefix_len: int) -> float:
        if prefix_len < self.min_prefix_len:
            return 0.0
        if prefix_len <= self.preferred_prefix_len:
            return 1.0
        overflow = prefix_len - self.preferred_prefix_len
        return max(0.45, 1.0 - 0.01 * overflow)

    def _reuse_probability(self, freq: float, observations: int, recent_support: float, recent_streak: int) -> float:
        observation_confidence = min(observations / max(self.min_observations + 2, 1), 1.0)
        streak_signal = min(recent_streak / 4.0, 1.0)
        score = (
            self.global_frequency_weight * min(freq, 1.0)
            + self.recent_support_weight * min(recent_support, 1.0)
            + self.recent_streak_weight * streak_signal
            + self.observation_weight * observation_confidence
        )
        probability = min(score * self.reuse_discount, self.max_reuse_probability)
        return max(probability, 0.0)

    def _expected_future_reuses(
        self,
        reuse_probability: float,
        recent_support: float,
        recent_streak: int,
        branching_factor: int,
        arrival_rate_rps: float,
    ) -> float:
        horizon_requests = max(arrival_rate_rps * self.reuse_horizon_s, 0.0)
        if horizon_requests <= 0.0:
            bootstrap_strength = (
                0.55 * min(recent_support, 1.0)
                + 0.25 * min(recent_streak / max(self.min_observations + 1, 1), 1.0)
                + 0.20 * min(reuse_probability, 1.0)
            )
            horizon_requests = max(
                self.bootstrap_horizon_requests * (0.75 + bootstrap_strength),
                reuse_probability,
            )
        streak_multiplier = 1.0 + 0.20 * min(recent_streak, 4)
        support_multiplier = 1.0 + 0.75 * min(recent_support, 1.0)
        branching_multiplier = 1.0 + self.branching_weight * min(max(branching_factor - 1, 0), 4)
        future_reuses = reuse_probability * horizon_requests * streak_multiplier * support_multiplier * branching_multiplier
        return max(min(future_reuses, horizon_requests * 1.5), reuse_probability)

    def _expected_saved_prefill_ms(self, prefix: Tuple[int, ...], expected_future_reuses: float) -> float:
        effective_len = min(len(prefix), self.max_prefix_len)
        reusable_cost = self.fixed_prefill_overhead_ms + (effective_len * self.ms_per_token)
        return float(expected_future_reuses * reusable_cost * self._length_quality(effective_len))

    def _expected_cost_ms(self, prefix: Tuple[int, ...], bank: TieredStateBank, target_tier: str) -> float:
        estimated_memory_mb = self._estimate_memory_mb(prefix)
        memory_pressure = bank.get_memory_bytes() / max(bank.max_memory_bytes, 1)
        pressure_multiplier = 1.0 + 1.25 * memory_pressure
        effective_len = min(len(prefix), self.max_prefix_len)
        prefill_cost = self.fixed_prefill_overhead_ms + (effective_len * self.ms_per_token)
        idle_fraction = self.gpu_idle_cost_fraction if target_tier == 'gpu' else self.idle_cost_fraction
        memory_cost = estimated_memory_mb * self.memory_penalty_per_mb
        return float((prefill_cost * idle_fraction + memory_cost) * pressure_multiplier)

    def _prune_dominated_candidates(self, candidates: List[dict]) -> List[dict]:
        kept: List[dict] = []
        sorted_candidates = sorted(
            candidates,
            key=lambda item: (
                -len(item['prefix']),
                -item['recent_support'],
                -item['reuse_probability'],
                -item['observations'],
            ),
        )
        for candidate in sorted_candidates:
            dominated = False
            for other in kept:
                if len(other['prefix']) <= len(candidate['prefix']):
                    continue
                if other['prefix'][: len(candidate['prefix'])] != candidate['prefix']:
                    continue
                if other['recent_support'] + 1e-9 < candidate['recent_support'] * self.dominance_tolerance:
                    continue
                if other['reuse_probability'] + 1e-9 < candidate['reuse_probability'] * self.dominance_tolerance:
                    continue
                if other['observations'] + 1e-9 < candidate['observations'] * 0.75:
                    continue
                dominated = True
                break
            if not dominated:
                kept.append(candidate)
        return kept

    def rank(self, bank: TieredStateBank, budget_k: int, prefer_gpu: bool = False) -> List[SpeculationDecision]:
        if budget_k <= 0:
            return []
        candidate_rows: List[dict] = []
        target_tier = 'gpu' if prefer_gpu else 'cpu'
        candidates = bank.get_candidate_stats(max_prefix_len=self.max_prefix_len, exclude_stored=True)
        arrival_rate_rps = bank.recent_arrival_rate()

        for prefix, freq, observations, branching_factor in candidates:
            if len(prefix) < self.min_prefix_len:
                continue
            if float(freq) < self.min_frequency or observations < self.min_observations:
                continue
            recent_support = bank.recent_prefix_support(prefix)
            recent_streak = bank.recent_prefix_streak(prefix)
            if recent_support < self.min_recent_support and observations < (self.min_observations + 2):
                continue
            reuse_probability = self._reuse_probability(float(freq), observations, recent_support, recent_streak)
            if reuse_probability <= 0.0:
                continue
            candidate_rows.append(
                {
                    'prefix': prefix,
                    'freq': float(freq),
                    'observations': observations,
                    'branching_factor': branching_factor,
                    'recent_support': recent_support,
                    'recent_streak': recent_streak,
                    'reuse_probability': reuse_probability,
                }
            )

        decisions: List[SpeculationDecision] = []
        for candidate in self._prune_dominated_candidates(candidate_rows):
            prefix = candidate['prefix']
            expected_future_reuses = self._expected_future_reuses(
                reuse_probability=candidate['reuse_probability'],
                recent_support=candidate['recent_support'],
                recent_streak=candidate['recent_streak'],
                branching_factor=candidate['branching_factor'],
                arrival_rate_rps=arrival_rate_rps,
            )
            expected_benefit = self._expected_saved_prefill_ms(prefix, expected_future_reuses)
            expected_cost = self._expected_cost_ms(prefix, bank, target_tier)
            expected_net = expected_benefit - expected_cost
            if expected_cost <= 0.0:
                continue
            if expected_net < self.min_expected_net_ms:
                continue
            if expected_benefit < self.benefit_cost_ratio * expected_cost:
                continue
            score = (
                expected_net / max(expected_cost, 1e-6)
                + 0.20 * candidate['reuse_probability']
                + 0.06 * min(max(candidate['branching_factor'] - 1, 0), 4)
                + 0.08 * self._length_quality(len(prefix))
            )
            if score < self.min_utility_score:
                continue
            decisions.append(
                SpeculationDecision(
                    prefix_tokens=prefix,
                    score=float(score),
                    target_tier=target_tier,
                    expected_benefit_ms=float(expected_benefit),
                    expected_cost_ms=float(expected_cost),
                )
            )

        decisions.sort(key=lambda d: (d.score, d.expected_benefit_ms, len(d.prefix_tokens)), reverse=True)
        return decisions[: min(budget_k, self.max_admissions_per_idle)]
