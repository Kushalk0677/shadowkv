from __future__ import annotations

from proactive_kv_cache.base_policy import ReusePlan
from proactive_kv_cache.backend.fake_backend import SemanticSafetySandbox


class NoLogitBackend:
    pass


def test_sandbox_rejects_distant_prefixes() -> None:
    plan = ReusePlan('semantic_partial', 4, 4, 0.2, 1.0, 0.1, 0.0, 0.9, 0.8, 'test')
    result = SemanticSafetySandbox().validate(NoLogitBackend(), (1, 2, 3, 4), (9, 8, 7, 6), plan)
    assert result.allowed is False
    assert result.divergence is not None


def test_sandbox_allows_identical_prefixes() -> None:
    plan = ReusePlan('semantic_partial', 4, 4, 0.2, 1.0, 0.1, 0.0, 0.9, 0.8, 'test')
    result = SemanticSafetySandbox().validate(NoLogitBackend(), (1, 2, 3, 4), (1, 2, 3, 4), plan)
    assert result.allowed is True
    assert result.divergence == 0.0
