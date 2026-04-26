from proactive_kv_cache.datasets import _apply_prompt_mode, list_datasets
from proactive_kv_cache.engines import ReactivePrefixCacheEngine
from proactive_kv_cache.models import FakeBackend
from proactive_kv_cache.policy import CostAwareSlackPolicy
from proactive_kv_cache.cache import TieredStateBank
from proactive_kv_cache.workload import make_synthetic_workload


def test_new_dataset_registry_entries_present():
    names = list_datasets()
    for expected in ('samsum', 'cnn_dailymail', 'banking77', 'ultrachat'):
        assert expected in names


def test_long_shared_prefix_variant_has_long_prompts():
    reqs = make_synthetic_workload('long_shared_prefix', 5, seed=7)
    assert len(reqs) == 5
    assert max(len(r.prompt) for r in reqs) > 250
    assert all(r.metadata and r.metadata.get('shared_prefix_text') for r in reqs)


def test_rag_variant_emits_rag_prompt_metadata():
    reqs = make_synthetic_workload('rag_long_context', 3, seed=5)
    assert len(reqs) == 3
    assert all(r.metadata and r.metadata.get('prompt_mode') == 'rag' for r in reqs)
    assert all(r.metadata and r.metadata.get('shared_prefix_text') for r in reqs)


def test_prompt_modes_add_shared_scaffolds():
    base_prompt = 'Instruction: do the thing\nAssistant response:'
    raw_prompt, raw_shared, raw_meta = _apply_prompt_mode('dolly', 'instruction', base_prompt, 'raw')
    templated_prompt, templated_shared, templated_meta = _apply_prompt_mode('dolly', 'instruction', base_prompt, 'templated')
    rag_prompt, rag_shared, rag_meta = _apply_prompt_mode('dolly', 'instruction', base_prompt, 'rag')
    assert raw_prompt == base_prompt
    assert raw_shared == ''
    assert templated_prompt.startswith(templated_shared)
    assert rag_prompt.startswith(rag_shared)
    assert len(rag_shared) > len(templated_shared) > 0


def test_scaffold_hint_tracks_only_the_full_shared_prefix():
    backend = FakeBackend()
    engine = ReactivePrefixCacheEngine(backend=backend)
    prompt = 'shared scaffold body user specific tail'
    tokens = backend.tokenize(prompt)
    metadata = {
        'prompt_mode': 'templated',
        'shared_prefix_hint_tokens': 5,
        'shared_prefix_text': 'shared scaffold body',
    }
    lengths = engine._tracked_prefix_lengths(tokens, metadata)
    assert lengths == [5]
    assert engine._reactive_prefix_len(tokens, metadata) == 5


def test_cost_policy_requires_observation_support():
    bank = TieredStateBank(max_memory_bytes=4096, min_match_length=3)
    for _ in range(2):
        bank.observe_query((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    policy = CostAwareSlackPolicy(min_prefix_len=8, min_observations=3, min_frequency=0.1, benefit_cost_ratio=0.1, min_expected_net_ms=0.1)
    assert policy.rank(bank, budget_k=3) == []


def test_fake_backend_capabilities_stay_external_kv():
    backend = FakeBackend()
    assert backend.supports_external_kv is True
    assert backend.supports_native_prefix_caching is False


def test_semantic_prompt_mode_rotates_paraphrased_scaffolds_and_sets_equivalence_key():
    base_prompt = 'Instruction: do the thing\nAssistant response:'
    p0, s0, m0 = _apply_prompt_mode('dolly', 'instruction', base_prompt, 'semantic', request_index=0)
    p1, s1, m1 = _apply_prompt_mode('dolly', 'instruction', base_prompt, 'semantic', request_index=1)
    assert p0.startswith(s0)
    assert p1.startswith(s1)
    assert s0 != s1
    assert m0['semantic_equivalence_key'] == m1['semantic_equivalence_key']
    assert m0['paraphrase_variant'] != m1['paraphrase_variant']


def test_synthetic_semantic_paraphrase_variant_rotates_without_exact_shared_prefix():
    reqs = make_synthetic_workload('semantic_paraphrase', 8, seed=11)
    assert len(reqs) == 8
    variants = {r.metadata.get('paraphrase_variant') for r in reqs if r.metadata}
    keys = {r.metadata.get('semantic_equivalence_key') for r in reqs if r.metadata}
    assert len(variants) > 1
    assert keys == {'synthetic_semantic_classification'}
    assert all(r.metadata and r.metadata.get('prompt_mode') == 'semantic' for r in reqs)
