from argparse import Namespace

import pytest

from experiments.run_benchmark import EXTERNAL_RUNTIME_BASELINE_NAMES, RUNTIME_BASELINE_NAMES, build_engine, list_engine_names
from proactive_kv_cache.engines import AdmissionControlledRuntimeCacheEngine, RuntimeNativeCacheEngine
from proactive_kv_cache.models import FakeBackend


def _args(**overrides):
    values = {
        "backend": "vllm",
        "include_runtime_baselines": True,
        "include_experimental": False,
        "include_semantic_ablations": False,
        "engines": None,
        "max_memory_mb": 16,
    }
    values.update(overrides)
    return Namespace(**values)


def test_runtime_baselines_are_additive_to_default_engine_list():
    names = list_engine_names(_args())
    for name in RUNTIME_BASELINE_NAMES:
        assert name in names
    assert "native_prefix_cache" in names
    assert "shadow_kv_plus_scaffold_only" not in names


def test_runtime_baseline_builds_plain_and_admission_variants():
    backend = FakeBackend()
    plain = build_engine(_args(), backend, "vllm_apc")
    controlled = build_engine(_args(), backend, "vllm_apc_shadowkv_plus")

    assert isinstance(plain, RuntimeNativeCacheEngine)
    assert isinstance(controlled, AdmissionControlledRuntimeCacheEngine)
    assert plain.name == "vllm_apc"
    assert controlled.name == "vllm_apc_shadowkv_plus"
    assert controlled.engine_metrics["admission_controller_enabled"] is True


def test_admission_controlled_runtime_baseline_serves_requests():
    backend = FakeBackend()
    engine = build_engine(_args(), backend, "vllm_apc_shadowkv_plus")
    metadata = {"prompt_mode": "templated", "shared_prefix_hint_tokens": 4}

    first = engine.serve_tokens(1, (1, 2, 3, 4, 5), metadata=metadata)
    second = engine.serve_tokens(2, (1, 2, 3, 4, 6), metadata=metadata)

    assert first.was_cache_hit is False
    assert second.request_id == 2
    assert engine.engine_metrics["admission_plans_total"] == 2


def test_external_runtime_baselines_are_not_in_process_placeholders():
    backend = FakeBackend()
    for name in EXTERNAL_RUNTIME_BASELINE_NAMES:
        with pytest.raises(ValueError, match="external runtime baseline"):
            build_engine(_args(), backend, name)
