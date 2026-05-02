from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

from .semantic import token_entropy

ReuseStrategy = Literal['bypass', 'exact', 'semantic_partial']


@dataclass
class ReusePlan:
    strategy: ReuseStrategy
    speculate_depth_tokens: int
    reusable_prefix_tokens: int
    layer_reuse_ratio: float
    expected_benefit_ms: float
    expected_cost_ms: float
    expected_waste_ms: float
    score: float
    confidence: float
    reason: str


class AdaptiveReuseController:
    """Waste-aware online controller for ShadowKV++.

    It converts prompt/workload features into a pre-execution plan. The score is
    a net utility objective:
        utility = expected_benefit - expected_cost - expected_waste
    """

    def __init__(
        self,
        min_utility_ms: float = 0.0,
        semantic_threshold: float = 0.58,
        max_layer_reuse_ratio: float = 0.50,
    ) -> None:
        self.min_utility_ms = min_utility_ms
        self.semantic_threshold = semantic_threshold
        self.max_layer_reuse_ratio = max_layer_reuse_ratio
        self._ewma_hit_rate = 0.0
        self._ewma_waste_ratio = 0.0
        self._alpha = 0.20

    def update_feedback(self, *, hit: bool, wasted_ratio: float) -> None:
        self._ewma_hit_rate = (1 - self._alpha) * self._ewma_hit_rate + self._alpha * (1.0 if hit else 0.0)
        self._ewma_waste_ratio = (1 - self._alpha) * self._ewma_waste_ratio + self._alpha * max(min(wasted_ratio, 1.0), 0.0)

    def plan(
        self,
        *,
        tokens: Tuple[int, ...],
        exact_match_len: int,
        semantic_similarity: float,
        semantic_prefix_len: int,
        shared_prefix_hint: int | None,
        full_ms_per_token: float,
        reuse_overhead_ms: float,
        metadata: Dict | None = None,
    ) -> ReusePlan:
        n = max(len(tokens), 1)
        prompt_mode = str((metadata or {}).get('prompt_mode', '')).lower()
        entropy = token_entropy(tokens)
        template_signal = 1.0 if shared_prefix_hint else 0.0
        if prompt_mode in {'templated', 'rag'}:
            template_signal = max(template_signal, 0.85)
        length_signal = min(n / 128.0, 1.0)
        entropy_signal = 1.0 - min(entropy / 8.0, 1.0) * 0.35
        health = 0.65 + 0.35 * self._ewma_hit_rate - 0.45 * self._ewma_waste_ratio
        health = max(min(health, 1.0), 0.05)

        if exact_match_len > 0:
            reusable = exact_match_len
            gross = reusable * max(full_ms_per_token, 0.05)
            suffix = max(n - reusable, 0)
            cost = reuse_overhead_ms + 0.02 * suffix
            waste = cost * max(self._ewma_waste_ratio, 0.02)
            score = gross - cost - waste
            confidence = max(0.5, min(0.98, 0.45 + 0.35 * (reusable / n) + 0.20 * template_signal)) * health
            layer_ratio = min(self.max_layer_reuse_ratio, 0.20 + 0.30 * confidence)
            return ReusePlan(
                strategy='exact' if score >= self.min_utility_ms else 'bypass',
                speculate_depth_tokens=reusable,
                reusable_prefix_tokens=reusable,
                layer_reuse_ratio=layer_ratio,
                expected_benefit_ms=gross,
                expected_cost_ms=cost,
                expected_waste_ms=waste,
                score=score,
                confidence=confidence,
                reason='exact_prefix_net_positive' if score >= self.min_utility_ms else 'exact_prefix_net_negative',
            )

        semantic_gate = semantic_similarity >= self.semantic_threshold and semantic_prefix_len > 0
        if semantic_gate:
            # Semantic prompt mode carries explicit equivalence-family metadata.
            # On real HF backends, approximate KV reuse is still blocked later;
            # this plan exists to measure safe, reviewable semantic opportunity.
            semantic_mode = prompt_mode == 'semantic'
            reusable = min(semantic_prefix_len, max(shared_prefix_hint or 0, int(n * (0.70 if semantic_mode else 0.55))))
            benefit_discount = 0.72 if semantic_mode else 0.55
            gross = reusable * max(full_ms_per_token, 0.05) * benefit_discount
            cost = reuse_overhead_ms + (0.04 if semantic_mode else 0.08) * max(n - reusable, 0)
            uncertainty = max(0.02, 1.0 - min(semantic_similarity, 0.98))
            waste = gross * uncertainty * (0.35 if semantic_mode else 1.0) + cost * max(self._ewma_waste_ratio, 0.03 if semantic_mode else 0.05)
            score = (gross - cost - waste) * health
            confidence = min(0.94 if semantic_mode else 0.90, semantic_similarity * ((0.70 if semantic_mode else 0.55) + 0.20 * template_signal + 0.10 * length_signal)) * entropy_signal * health
            layer_ratio = min(self.max_layer_reuse_ratio, max(0.10, (0.55 if semantic_mode else 0.45) * confidence))
            return ReusePlan(
                strategy='semantic_partial' if score >= self.min_utility_ms else 'bypass',
                speculate_depth_tokens=reusable,
                reusable_prefix_tokens=reusable,
                layer_reuse_ratio=layer_ratio,
                expected_benefit_ms=gross,
                expected_cost_ms=cost,
                expected_waste_ms=waste,
                score=score,
                confidence=confidence,
                reason='semantic_partial_net_positive' if score >= self.min_utility_ms else 'semantic_partial_net_negative',
            )

        return ReusePlan(
            strategy='bypass',
            speculate_depth_tokens=0,
            reusable_prefix_tokens=0,
            layer_reuse_ratio=0.0,
            expected_benefit_ms=0.0,
            expected_cost_ms=0.0,
            expected_waste_ms=0.0,
            score=-reuse_overhead_ms,
            confidence=0.0,
            reason='no_reuse_signal',
        )
