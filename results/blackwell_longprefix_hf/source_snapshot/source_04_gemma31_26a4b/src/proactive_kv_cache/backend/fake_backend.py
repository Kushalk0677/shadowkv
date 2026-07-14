from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..base_policy import ReusePlan
from ..config_loader import CONFIG, RuntimeConfig


@dataclass
class SemanticSafetyResult:
    allowed: bool
    divergence: float | None
    reason: str


class SemanticSafetySandbox:
    """Validates a semantic KV candidate before the real backend may reuse it."""

    def __init__(self, config: RuntimeConfig = CONFIG) -> None:
        self.config = config

    def validate(self, backend, current_tokens: Tuple[int, ...], candidate_tokens: Tuple[int, ...], plan: ReusePlan) -> SemanticSafetyResult:
        if not bool(self.config.get('semantic.sandbox.enabled', True)):
            return SemanticSafetyResult(False, None, 'semantic_sandbox_disabled')
        probe_len = min(max(int(plan.reusable_prefix_tokens), 1), len(current_tokens), len(candidate_tokens))
        if probe_len <= 0:
            return SemanticSafetyResult(False, None, 'empty_probe')
        tolerance = float(self.config.get('semantic.sandbox.max_divergence', 0.08))
        top_k = int(self.config.get('semantic.sandbox.top_k', 32))
        distance = None
        if hasattr(backend, 'logit_guard_distance'):
            try:
                distance = backend.logit_guard_distance(current_tokens[:probe_len], candidate_tokens[:probe_len], top_k=top_k)
            except Exception:
                distance = None
        if distance is None:
            current = set(current_tokens[:probe_len])
            candidate = set(candidate_tokens[:probe_len])
            union = max(len(current | candidate), 1)
            distance = 1.0 - (len(current & candidate) / union)
        divergence = max(float(distance), 0.0)
        return SemanticSafetyResult(
            allowed=divergence <= tolerance,
            divergence=divergence,
            reason='semantic_sandbox_pass' if divergence <= tolerance else 'semantic_sandbox_reject',
        )
