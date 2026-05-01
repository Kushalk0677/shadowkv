from __future__ import annotations

from typing import Dict, Tuple

from .base_policy import PolicyController, ReusePlan
from .config_loader import CONFIG, RuntimeConfig
from .utility import AdmissionEvent, UtilityModel


class UtilityPolicyController(PolicyController):
    def __init__(
        self,
        *,
        min_utility_ms: float | None = None,
        semantic_threshold: float | None = None,
        max_layer_reuse_ratio: float = 0.50,
        config: RuntimeConfig = CONFIG,
    ) -> None:
        self.config = config
        self.min_utility_ms = float(config.get('policy.utility.util_min_ms', 0.0) if min_utility_ms is None else min_utility_ms)
        self.semantic_threshold = float(config.get('policy.utility.semantic_threshold', 0.58) if semantic_threshold is None else semantic_threshold)
        self.max_layer_reuse_ratio = float(max_layer_reuse_ratio)
        self._ewma_hit_rate = 0.0
        self._ewma_waste_ratio = 0.0
        self._alpha = float(config.get('policy.health.ewma_alpha', 0.20))
        self.utility = UtilityModel(config)
        self.last_breakdown = None

    def update_feedback(self, *, hit: bool, wasted_ratio: float) -> None:
        alpha = float(self.config.get('policy.health.ewma_alpha', self._alpha))
        self._ewma_hit_rate = (1 - alpha) * self._ewma_hit_rate + alpha * (1.0 if hit else 0.0)
        self._ewma_waste_ratio = (1 - alpha) * self._ewma_waste_ratio + alpha * max(min(float(wasted_ratio), 1.0), 0.0)

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
        event = AdmissionEvent(
            tokens=tokens,
            exact_match_len=exact_match_len,
            semantic_similarity=semantic_similarity,
            semantic_prefix_len=semantic_prefix_len,
            shared_prefix_hint=shared_prefix_hint,
            full_ms_per_token=full_ms_per_token,
            reuse_overhead_ms=reuse_overhead_ms,
            ewma_hit_rate=self._ewma_hit_rate,
            ewma_waste_ratio=self._ewma_waste_ratio,
            max_layer_reuse_ratio=self.max_layer_reuse_ratio,
            min_utility_ms=self.min_utility_ms,
            semantic_threshold=self.semantic_threshold,
            metadata=metadata,
        )
        breakdown = self.utility.admission(event)
        self.last_breakdown = breakdown
        return ReusePlan(
            strategy=breakdown.strategy,
            speculate_depth_tokens=breakdown.speculate_depth_tokens,
            reusable_prefix_tokens=breakdown.reusable_prefix_tokens,
            layer_reuse_ratio=breakdown.layer_reuse_ratio,
            expected_benefit_ms=breakdown.benefit_ms,
            expected_cost_ms=breakdown.cost_ms,
            expected_waste_ms=breakdown.waste_ms,
            score=breakdown.net_utility_ms,
            confidence=breakdown.confidence,
            reason=breakdown.reason,
        )
