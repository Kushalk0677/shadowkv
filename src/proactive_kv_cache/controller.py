from __future__ import annotations

from typing import Dict, Tuple

from .base_policy import ReusePlan, ReuseStrategy
from .config_loader import CONFIG
from .utility_policy import UtilityPolicyController


class AdaptiveReuseController:
    """Compatibility wrapper around the pluggable utility policy."""

    def __init__(
        self,
        min_utility_ms: float | None = None,
        semantic_threshold: float | None = None,
        max_layer_reuse_ratio: float = 0.50,
    ) -> None:
        self.policy = UtilityPolicyController(
            min_utility_ms=CONFIG.get('policy.utility.util_min_ms', 0.0) if min_utility_ms is None else min_utility_ms,
            semantic_threshold=CONFIG.get('policy.utility.semantic_threshold', 0.58) if semantic_threshold is None else semantic_threshold,
            max_layer_reuse_ratio=max_layer_reuse_ratio,
            config=CONFIG,
        )

    @property
    def min_utility_ms(self) -> float:
        return self.policy.min_utility_ms

    @property
    def semantic_threshold(self) -> float:
        return self.policy.semantic_threshold

    @property
    def max_layer_reuse_ratio(self) -> float:
        return self.policy.max_layer_reuse_ratio

    @property
    def last_breakdown(self):
        return self.policy.last_breakdown

    def update_feedback(self, *, hit: bool, wasted_ratio: float) -> None:
        self.policy.update_feedback(hit=hit, wasted_ratio=wasted_ratio)

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
        return self.policy.plan(
            tokens=tokens,
            exact_match_len=exact_match_len,
            semantic_similarity=semantic_similarity,
            semantic_prefix_len=semantic_prefix_len,
            shared_prefix_hint=shared_prefix_hint,
            full_ms_per_token=full_ms_per_token,
            reuse_overhead_ms=reuse_overhead_ms,
            metadata=metadata,
        )


__all__ = ['AdaptiveReuseController', 'ReusePlan', 'ReuseStrategy']
