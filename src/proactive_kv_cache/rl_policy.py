from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from .base_policy import PolicyController, ReusePlan


class LearningPolicyController(PolicyController):
    """Placeholder learning policy.

    It intentionally defaults to bypass unless a future learner file defines a
    supported policy. This gives experiments a stable plug-in point without
    hiding unvalidated decisions behind the current controller.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.model_spec = {}
        if self.model_path and self.model_path.exists():
            self.model_spec = json.loads(self.model_path.read_text(encoding='utf-8'))

    def update_feedback(self, *, hit: bool, wasted_ratio: float) -> None:
        return None

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
        return ReusePlan(
            strategy='bypass',
            speculate_depth_tokens=0,
            reusable_prefix_tokens=0,
            layer_reuse_ratio=0.0,
            expected_benefit_ms=0.0,
            expected_cost_ms=0.0,
            expected_waste_ms=0.0,
            score=-float(reuse_overhead_ms),
            confidence=0.0,
            reason='learning_policy_placeholder',
        )
