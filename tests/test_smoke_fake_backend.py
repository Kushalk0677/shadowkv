from proactive_kv_cache.engines import (
    FrequencySpeculativeEngine,
    NoCacheEngine,
    ReactivePrefixCacheEngine,
    ShadowKVEngine,
    compare_named_runs,
    maybe_shutdown,
)
from proactive_kv_cache.models import load_backend
from proactive_kv_cache.workload import make_synthetic_workload


def test_fake_backend_end_to_end():
    backend = load_backend('fake')
    requests = make_synthetic_workload('moderate_skew', n_requests=10, seed=7)
    engines = [
        NoCacheEngine(backend=backend, max_memory_mb=16),
        ReactivePrefixCacheEngine(backend=backend, max_memory_mb=16),
        FrequencySpeculativeEngine(backend=backend, max_memory_mb=16, speculative_k=2, idle_threshold_ms=1.0),
        ShadowKVEngine(backend=backend, max_memory_mb=16, speculative_k=2, idle_threshold_ms=1.0, enable_gpu_tier=False),
    ]
    for engine in engines:
        for req in requests:
            tokens = backend.tokenize(req.prompt)
            engine.serve_tokens(req.request_id, tokens)
        maybe_shutdown(engine)
    summary = compare_named_runs(engines)
    assert 'no_cache' in summary and 'shadow_kv' in summary
    assert summary['no_cache']['requests_seen'] == len(requests)
    assert summary['shadow_kv']['requests_seen'] == len(requests)
    assert summary['no_cache']['hit_rate'] == 0.0
    assert summary['reactive_prefix_cache']['store_attempts'] > 0
    assert summary['shadow_kv']['wasted_compute_ratio'] <= 1.0
