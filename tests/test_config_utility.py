from __future__ import annotations

from proactive_kv_cache.config_loader import CONFIG
from proactive_kv_cache.controller import AdaptiveReuseController
from proactive_kv_cache.prefix_gate import breakeven_prefix_len
from proactive_kv_cache.utility import AdmissionEvent, UtilityModel


def test_config_loads_version() -> None:
    assert CONFIG.version
    assert CONFIG.get('policy.utility.semantic_threshold') is not None


def test_prefix_gate_respects_minimum() -> None:
    assert breakeven_prefix_len() >= int(CONFIG.get('hardware.min_prefix_len', 1))


def test_utility_model_admits_exact_positive_prefix() -> None:
    model = UtilityModel()
    result = model.admission(
        AdmissionEvent(
            tokens=tuple(range(32)),
            exact_match_len=24,
            semantic_similarity=0.0,
            semantic_prefix_len=0,
            shared_prefix_hint=24,
            full_ms_per_token=1.0,
            reuse_overhead_ms=1.0,
            ewma_hit_rate=0.5,
            ewma_waste_ratio=0.0,
            max_layer_reuse_ratio=0.55,
            min_utility_ms=0.0,
            semantic_threshold=0.58,
            metadata={'prompt_mode': 'templated'},
        )
    )
    assert result.strategy == 'exact'
    assert result.net_utility_ms > 0


def test_controller_compatibility_wrapper_returns_plan() -> None:
    controller = AdaptiveReuseController()
    plan = controller.plan(
        tokens=tuple(range(16)),
        exact_match_len=8,
        semantic_similarity=0.0,
        semantic_prefix_len=0,
        shared_prefix_hint=8,
        full_ms_per_token=0.7,
        reuse_overhead_ms=1.0,
        metadata={'prompt_mode': 'templated'},
    )
    assert plan.reusable_prefix_tokens == 8
    assert plan.strategy in {'exact', 'bypass'}
