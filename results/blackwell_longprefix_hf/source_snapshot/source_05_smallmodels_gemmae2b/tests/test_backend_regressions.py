import os
from argparse import Namespace

import pytest

import experiments.run_benchmark as benchmark
from proactive_kv_cache.models import FakeBackend, HuggingFaceBackend, load_backend


def test_fake_backend_uses_persistent_vocabulary_across_prompts():
    backend = FakeBackend()

    alpha = backend.tokenize('shared prefix alpha')
    beta = backend.tokenize('shared prefix beta')
    unrelated = backend.tokenize('different prefix beta')

    assert alpha[:2] == beta[:2]
    assert alpha[:2] != unrelated[:2]
    assert alpha[-1] != beta[-1]


def test_fake_backend_cached_prefill_extends_existing_state():
    backend = FakeBackend()
    # Disable compression for this test to inspect tokens
    backend._disable_compression = True
    prefix = backend.tokenize('shared prefix')
    suffix = backend.tokenize('new suffix')
    
    prefix_out = backend.prefill(prefix)
    combined_out = backend.prefill(suffix, past_key_values=prefix_out.kv_cache)

    assert combined_out.kv_cache['tokens'] == prefix + suffix
    assert combined_out.kv_cache['prefix_len'] == len(prefix) + len(suffix)
    assert combined_out.memory_bytes > prefix_out.memory_bytes


def test_fake_backend_slices_compressed_kv_by_tokens_not_bytes():
    backend = FakeBackend()
    tokens = backend.tokenize('shared prefix compressed cache suffix')

    prefill = backend.prefill(tokens)
    sliced = backend.slice_past_key_values(prefill.kv_cache, 3)
    combined = backend.prefill(tokens[3:], past_key_values=sliced)

    assert sliced['tokens'] == tokens[:3]
    assert sliced['compressed'] is False
    assert combined.prepared_past_length == 3
    assert combined.kv_cache['prefix_len'] == len(tokens)


def test_hf_cache_preparation_isolates_canonical_tensors():
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')

    backend = object.__new__(HuggingFaceBackend)
    backend.device = 'cpu'
    backend.model_name = 'test-model'
    backend.max_positions = 32
    original = ((torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 4, 2)),)

    prepared = backend._isolate_past_key_values(original)
    prepared[0][0].add_(1)
    prepared[0][1].add_(1)

    assert torch.count_nonzero(original[0][0]).item() == 0
    assert torch.count_nonzero(original[0][1]).item() == 0


def test_shadowkv_calibration_reuses_loaded_backend(monkeypatch):
    backend = FakeBackend()
    backend.device = 'cuda'
    args = Namespace(
        policy_preset='balanced',
        _shadowkv_policy_calibration_base=None,
        device='cuda',
        resolved_prompt_mode='semantic',
        max_memory_mb=16,
        speculative_k=2,
        idle_threshold_ms=30.0,
    )

    def fail_if_loaded(_args):
        raise AssertionError('calibration must not load a second backend')

    monkeypatch.setattr(benchmark, 'load_backend_from_args', fail_if_loaded)
    engine = benchmark.build_engine(args, backend, 'shadow_kv')

    assert engine.backend is backend
    assert args._shadowkv_policy_calibration_base is not None


@pytest.mark.skipif(
    os.environ.get('RUN_HF_KV_CORRECTNESS') != '1',
    reason='set RUN_HF_KV_CORRECTNESS=1 to run the slow Hugging Face KV correctness check',
)
def test_hf_cached_prefill_matches_full_prefill_cache_shape_for_tiny_model():
    pytest.importorskip('torch')
    pytest.importorskip('transformers')

    backend = load_backend('hf', model_name='sshleifer/tiny-gpt2', device='cpu')
    tokens = backend.tokenize('System: answer carefully.\nQuestion: What is cached prefill?')
    prefix_len = max(3, len(tokens) // 2)

    full = backend.prefill(tokens)
    prefix = backend.prefill(tokens[:prefix_len])
    cached = backend.prefill(tokens[prefix_len:], past_key_values=prefix.kv_cache)

    assert backend._past_length(full.kv_cache) == backend._past_length(cached.kv_cache)
    assert full.memory_bytes == cached.memory_bytes
