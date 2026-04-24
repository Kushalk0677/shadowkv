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
        for prefix, freq, observations in candidates:
            if len(prefix) < self.min_prefix_len or freq < self.min_frequency or observations < self.min_observations:
                continue
            effective_len = min(len(prefix), self.max_prefix_len)
            confidence = min(observations / 6.0, 1.0)
            score = float(freq) * confidence * (1.0 + 0.08 * effective_len)
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
    """Rank prefixes that look worth precomputing during idle time."""

    def __init__(
        self,
        min_frequency: float = 0.18,
        token_benefit_ms: float = 1.2,
        speculation_penalty_ms: float = 5.0,
        memory_penalty_per_mb: float = 0.9,
        gpu_bonus_ms: float = 4.0,
        benefit_cost_ratio: float = 1.35,
        max_admissions_per_idle: int = 1,
        min_prefix_len: int = 12,
        preferred_prefix_len: int = 48,
        max_prefix_len: int = 128,
        min_observations: int = 3,
        min_expected_net_ms: float = 12.0,
    ):
        self.min_frequency = min_frequency
        self.token_benefit_ms = token_benefit_ms
        self.speculation_penalty_ms = speculation_penalty_ms
        self.memory_penalty_per_mb = memory_penalty_per_mb
        self.gpu_bonus_ms = gpu_bonus_ms
        self.benefit_cost_ratio = benefit_cost_ratio
        self.max_admissions_per_idle = max_admissions_per_idle
        self.min_prefix_len = min_prefix_len
        self.preferred_prefix_len = preferred_prefix_len
        self.max_prefix_len = max_prefix_len
        self.min_observations = min_observations
        self.min_expected_net_ms = min_expected_net_ms

    def _estimate_memory_mb(self, prefix: Tuple[int, ...]) -> float:
        return max(len(prefix) * 0.0015, 0.002)

    def _template_bonus(self, prefix_len: int, observations: int) -> float:
        if prefix_len < self.min_prefix_len:
            return -999.0
        bonus = 0.0
        if prefix_len <= self.preferred_prefix_len:
            bonus += 6.0
        elif prefix_len <= self.max_prefix_len:
            bonus += 3.0
        else:
            bonus -= 8.0
        bonus += min(observations, 8) * 0.8
        return bonus

    def _expected_benefit_ms(self, prefix: Tuple[int, ...], freq: float, observations: int, target_tier: str) -> float:
        effective_len = min(len(prefix), self.max_prefix_len)
        confidence = min(observations / max(self.min_observations + 2, 1), 1.0)
        benefit = freq * confidence * effective_len * self.token_benefit_ms
        benefit += self._template_bonus(len(prefix), observations)
        if target_tier == 'gpu':
            benefit += self.gpu_bonus_ms
        return float(benefit)

    def _expected_cost_ms(self, prefix: Tuple[int, ...], bank: TieredStateBank, target_tier: str) -> float:
        estimated_memory_mb = self._estimate_memory_mb(prefix)
        memory_pressure = bank.get_memory_bytes() / max(bank.max_memory_bytes, 1)
        pressure_multiplier = 1.0 + 1.5 * memory_pressure
        length_penalty = max(len(prefix) - self.preferred_prefix_len, 0) * 0.08
        tier_penalty = 1.5 if target_tier == 'gpu' else 0.0
        return float((self.speculation_penalty_ms + estimated_memory_mb * self.memory_penalty_per_mb + length_penalty + tier_penalty) * pressure_multiplier)

    def rank(self, bank: TieredStateBank, budget_k: int, prefer_gpu: bool = False) -> List[SpeculationDecision]:
        decisions: List[SpeculationDecision] = []
        target_tier = 'gpu' if prefer_gpu else 'cpu'
        candidates = bank.get_candidate_stats(max_prefix_len=self.max_prefix_len, exclude_stored=True)

        for prefix, freq, observations in candidates:
            if len(prefix) < self.min_prefix_len:
                continue
            if float(freq) < self.min_frequency or observations < self.min_observations:
                continue
            expected_benefit = self._expected_benefit_ms(prefix, float(freq), observations, target_tier)
            expected_cost = self._expected_cost_ms(prefix, bank, target_tier)
            expected_net = expected_benefit - expected_cost
            if expected_cost <= 0.0:
                continue
            if expected_net < self.min_expected_net_ms:
                continue
            if expected_benefit < self.benefit_cost_ratio * expected_cost:
                continue
            score = (expected_net / expected_cost) + (0.004 * min(len(prefix), self.preferred_prefix_len)) + min(observations, 10) * 0.02
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
