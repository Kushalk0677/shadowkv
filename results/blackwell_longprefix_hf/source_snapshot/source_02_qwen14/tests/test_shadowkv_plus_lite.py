from proactive_kv_cache.engines import ShadowKVPlusLiteEngine, summarize_engine
from proactive_kv_cache.models import FakeBackend


def test_shadowkv_plus_lite_reuses_long_scaffold_without_policy_planner():
    backend = FakeBackend()
    engine = ShadowKVPlusLiteEngine(backend=backend, min_reuse_prefix_tokens=8)
    shared = ' '.join(f'shared{i}' for i in range(16))
    meta = {
        'prompt_mode': 'templated',
        'shared_prefix_text': shared,
        'shared_prefix_hint_tokens': 16,
    }
    first = backend.tokenize(f'{shared} tail one')
    second = backend.tokenize(f'{shared} tail two')

    r1 = engine.serve_tokens(1, first, metadata=dict(meta))
    r2 = engine.serve_tokens(2, second, metadata=dict(meta))
    engine.shutdown()

    assert r1.was_cache_hit is False
    assert r2.was_cache_hit is True
    assert r2.matched_prefix_length == 16
    assert engine.engine_metrics['lite_fast_path_total'] >= 1
    assert engine.engine_metrics['policy_plans_total'] == 0
    assert engine.engine_metrics['semantic_queries_total'] == 0


def test_shadowkv_plus_lite_bypasses_short_prefixes():
    backend = FakeBackend()
    engine = ShadowKVPlusLiteEngine(backend=backend, min_reuse_prefix_tokens=12)
    shared = 'short shared prefix'
    meta = {
        'prompt_mode': 'templated',
        'shared_prefix_text': shared,
        'shared_prefix_hint_tokens': 3,
    }
    first = backend.tokenize(f'{shared} one')
    second = backend.tokenize(f'{shared} two')

    engine.serve_tokens(1, first, metadata=dict(meta))
    r2 = engine.serve_tokens(2, second, metadata=dict(meta))
    engine.shutdown()

    assert r2.was_cache_hit is False
    assert engine.engine_metrics['store_successes'] == 0
