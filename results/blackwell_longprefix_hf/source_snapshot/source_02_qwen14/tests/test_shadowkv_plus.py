import json
from pathlib import Path

from proactive_kv_cache.base_policy import ReusePlan
from proactive_kv_cache.controller import AdaptiveReuseController
from proactive_kv_cache.engines import ShadowKVPlusEngine
from proactive_kv_cache.models import FakeBackend
from proactive_kv_cache.policy_learning import RunFeatureRow, learn_shadowkv_plus_thresholds, load_feature_rows
from proactive_kv_cache.semantic import SemanticKVIndex, token_entropy


class RecordingFakeBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.last_prepared_past_length = None

    def prefill(self, tokens, past_key_values=None):
        result = super().prefill(tokens, past_key_values=past_key_values)
        self.last_prepared_past_length = result.prepared_past_length
        return result


def test_semantic_index_returns_neighbour_for_related_token_shape():
    idx = SemanticKVIndex(dims=64, max_entries=10)
    idx.add((10, 11, 12, 13, 14, 15))
    matches = idx.query((10, 11, 12, 99, 98), min_similarity=0.05)
    assert matches
    assert matches[0].similarity > 0


def test_controller_bypasses_when_no_reuse_signal():
    c = AdaptiveReuseController()
    plan = c.plan(
        tokens=(1, 2, 3, 4, 5),
        exact_match_len=0,
        semantic_similarity=0.0,
        semantic_prefix_len=0,
        shared_prefix_hint=None,
        full_ms_per_token=2.0,
        reuse_overhead_ms=4.0,
        metadata={},
    )
    assert plan.strategy == 'bypass'
    assert plan.expected_waste_ms == 0.0


def test_controller_exact_prefix_positive_plan():
    c = AdaptiveReuseController()
    plan = c.plan(
        tokens=tuple(range(64)),
        exact_match_len=40,
        semantic_similarity=0.0,
        semantic_prefix_len=0,
        shared_prefix_hint=40,
        full_ms_per_token=2.0,
        reuse_overhead_ms=2.0,
        metadata={'prompt_mode': 'templated'},
    )
    assert plan.strategy == 'exact'
    assert plan.speculate_depth_tokens == 40
    assert plan.layer_reuse_ratio > 0


def test_shadowkv_plus_records_policy_metrics():
    backend = FakeBackend()
    engine = ShadowKVPlusEngine(backend=backend)
    requests = [
        tuple([1, 2, 3, 4, 5, 6, i]) for i in range(10)
    ]
    for i, tokens in enumerate(requests):
        engine.serve_tokens(i, tokens, metadata={'prompt_mode': 'templated', 'shared_prefix_hint_tokens': 6})
    assert engine.engine_metrics['policy_plans_total'] == len(requests)
    assert engine.engine_metrics['policy_exact_total'] == len(requests) - 1
    assert engine.engine_metrics['fast_exact_path_hits'] == len(requests) - 1
    assert engine.engine_metrics['semantic_queries_total'] == 0
    assert engine.engine_metrics['semantic_queries_skipped_total'] >= 1
    assert 'policy_net_utility_ms' in engine.engine_metrics


def test_shadowkv_plus_warms_templated_scaffold_before_exact_reuse():
    backend = FakeBackend()
    engine = ShadowKVPlusEngine(backend=backend)
    shared = ' '.join(f'shared{i}' for i in range(24))
    metadata = {
        'prompt_mode': 'templated',
        'shared_prefix_hint_tokens': 24,
        'shared_prefix_text': shared,
    }
    requests = [
        f'{shared} user tail one',
        f'{shared} user tail two',
        f'{shared} user tail three',
        f'{shared} user tail four',
    ]

    for i, text in enumerate(requests):
        engine.serve_tokens(i, backend.tokenize(text), metadata=dict(metadata))

    assert engine.engine_metrics['scaffold_bypass_store_successes'] == 1
    assert engine.engine_metrics['policy_bypass_total'] == 1
    assert engine.engine_metrics['policy_exact_total'] == len(requests) - 1
    assert engine.engine_metrics['fast_exact_path_hits'] == len(requests) - 1
    assert engine.engine_metrics['reuse_successes'] == len(requests) - 1
    assert engine.engine_metrics['auto_disabled_reason'] is None


def test_shadowkv_plus_reuses_cacheable_slice_of_oversized_scaffold():
    backend = FakeBackend()
    engine = ShadowKVPlusEngine(backend=backend)
    cacheable = engine.tuning.max_cacheable_prefix_tokens
    shared = tuple(range(1, cacheable + 65))
    first = shared + tuple(range(10_000, 10_000 + cacheable * 4))
    second = shared + tuple(range(20_000, 20_000 + cacheable * 4))
    metadata = {
        'prompt_mode': 'semantic',
        'shared_prefix_hint_tokens': len(shared),
        'shared_prefix_text': 'oversized shared scaffold',
        'semantic_equivalence_key': 'oversized-scaffold',
    }

    engine.serve_tokens(0, first, metadata=dict(metadata))
    result = engine.serve_tokens(1, second, metadata=dict(metadata))

    assert result.was_cache_hit is True
    assert result.matched_prefix_length == cacheable
    assert engine.engine_metrics['fast_exact_path_hits'] == 1
    assert engine.engine_metrics['reuse_successes'] == 1


def test_shadowkv_plus_exact_fast_path_skips_semantic_and_policy_planner():
    backend = FakeBackend()
    engine = ShadowKVPlusEngine(backend=backend)
    shared = tuple(range(1, 17))
    first = shared + (101,)
    second = shared + (202,)
    metadata = {'prompt_mode': 'rag', 'shared_prefix_hint_tokens': len(shared)}

    engine.serve_tokens(0, first, metadata=metadata)
    result = engine.serve_tokens(1, second, metadata=metadata)

    assert result.was_cache_hit is True
    assert result.matched_prefix_length == len(shared)
    assert engine.engine_metrics['policy_plans_total'] == 2
    assert engine.engine_metrics['semantic_queries_total'] == 0
    assert engine.engine_metrics['semantic_queries_skipped_total'] >= 1
    assert engine.engine_metrics['fast_exact_path_hits'] == 1


def test_policy_learning_returns_thresholds():
    rows = [
        RunFeatureRow('a', 'shadow_kv_plus', 'synthetic', 'x', 'templated', 10, 10.0, 1.2, 0.5, 0.1, 50, 50),
        RunFeatureRow('b', 'shadow_kv_plus', 'synthetic', 'y', 'raw', 10, 20.0, 0.9, 0.0, 0.8, 0, 100),
    ]
    learned = learn_shadowkv_plus_thresholds(rows)
    assert learned['n_training_rows'] == 2
    assert 0.0 <= learned['estimated_accuracy'] <= 1.0


def test_token_entropy_zero_for_empty():
    assert token_entropy(()) == 0.0


def test_semantic_index_equivalence_key_boosts_paraphrase_match():
    idx = SemanticKVIndex(dims=64, max_entries=10)
    idx.add((101, 102, 103, 104), semantic_key='classify:ag_news')
    matches = idx.query((201, 202, 203, 204), min_similarity=0.80, semantic_key='classify:ag_news')
    assert matches
    assert matches[0].similarity >= 0.90


def test_shadowkv_plus_records_semantic_partial_opportunity_on_fake_backend():
    backend = FakeBackend()
    engine = ShadowKVPlusEngine(backend=backend, semantic_similarity_threshold=0.80)
    metadata = {
        'prompt_mode': 'semantic',
        'semantic_equivalence_key': 'semantic_task_classification:ag_news',
        'shared_prefix_hint_tokens': 24,
    }
    base_a = tuple(range(10, 34))
    base_b = tuple(range(110, 134))
    requests = [
        base_a + (1000 + i,) if i % 2 == 0 else base_b + (1000 + i,)
        for i in range(12)
    ]
    for i, tokens in enumerate(requests):
        engine.serve_tokens(i, tokens, metadata=metadata)
    assert engine.engine_metrics['policy_plans_total'] == len(requests)
    assert engine.engine_metrics['policy_semantic_partial_total'] > 0
    assert engine.engine_metrics['semantic_opportunity_plans_total'] > 0


def test_shadowkv_plus_slices_semantic_partial_kv_to_reusable_prefix():
    backend = RecordingFakeBackend()
    engine = ShadowKVPlusEngine(
        backend=backend,
        semantic_ablation_mode='early_layer',
        early_layer_reuse_ratio=0.5,
    )
    semantic_key = tuple(range(1, 9))
    current_tokens = tuple(range(101, 113))
    stored = backend.prefill(semantic_key)
    assert engine.bank.store(
        semantic_key,
        stored.kv_cache,
        stored.latency_ms,
        stored.memory_bytes,
        is_speculative=False,
    )
    engine._current_semantic_match = {'semantic_similarity': 0.95}
    plan = ReusePlan(
        strategy='semantic_partial',
        speculate_depth_tokens=8,
        reusable_prefix_tokens=8,
        layer_reuse_ratio=1.0,
        expected_benefit_ms=10.0,
        expected_cost_ms=1.0,
        expected_waste_ms=0.0,
        score=9.0,
        confidence=0.9,
        reason='test',
    )

    result = engine._partial_semantic_reuse(1, current_tokens, semantic_key, plan)

    assert result is not None
    assert result.matched_prefix_length == 4
    assert backend.last_prepared_past_length == 4


def test_shadowkv_plus_prefers_gpu_store_tier_for_hf_cuda_backend():
    backend = FakeBackend(device='cuda:0')
    backend.backend_name = 'hf'
    engine = ShadowKVPlusEngine(backend=backend)

    assert engine._default_shadowkv_plus_store_tier() == 'gpu'


def test_policy_learning_loads_summary_hit_rate_key():
    path = Path(__file__).with_name('_policy_learning_summary_fixture.json')
    try:
        path.write_text(json.dumps({
            'shadow_kv_plus': {
                'mean_latency_ms': 10.0,
                'speedup_vs_no_cache_mean': 1.25,
                'hit_rate': 0.42,
                'wasted_compute_ratio': 0.1,
                'reused_prefix_tokens_total': 42,
                'recompute_tokens_total': 58,
            },
            'config': {
                'workload': 'synthetic',
                'variant': 'high_skew',
                'prompt_mode': 'templated',
                'n_requests': 8,
            },
        }))

        rows = load_feature_rows([path])

        assert len(rows) == 1
        assert rows[0].cache_hit_rate == 0.42
    finally:
        path.unlink(missing_ok=True)
