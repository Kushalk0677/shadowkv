from proactive_kv_cache.controller import AdaptiveReuseController
from proactive_kv_cache.engines import ShadowKVPlusEngine
from proactive_kv_cache.models import FakeBackend
from proactive_kv_cache.policy_learning import RunFeatureRow, learn_shadowkv_plus_thresholds
from proactive_kv_cache.semantic import SemanticKVIndex, token_entropy


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
    assert engine.engine_metrics['semantic_queries_total'] == len(requests)
    assert 'policy_net_utility_ms' in engine.engine_metrics


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
        'shared_prefix_hint_tokens': 6,
    }
    requests = [
        tuple([10, 11, 12, 13, 14, 15, 100 + i]) if i % 2 == 0 else tuple([20, 21, 22, 23, 24, 25, 100 + i])
        for i in range(12)
    ]
    for i, tokens in enumerate(requests):
        engine.serve_tokens(i, tokens, metadata=metadata)
    assert engine.engine_metrics['policy_plans_total'] == len(requests)
    assert engine.engine_metrics['policy_semantic_partial_total'] > 0
    assert engine.engine_metrics['semantic_opportunity_plans_total'] > 0
