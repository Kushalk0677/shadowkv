from proactive_kv_cache.engines import ReactivePrefixCacheEngine, ShadowKVEngine
from proactive_kv_cache.models import FakeBackend


class FailingCacheBackend(FakeBackend):
    def __init__(self):
        super().__init__(device='cpu')
        self.fail_once = True

    def prefill(self, tokens, past_key_values=None):
        if past_key_values is not None and self.fail_once:
            self.fail_once = False
            raise IndexError('simulated cache reuse failure')
        return super().prefill(tokens, past_key_values=past_key_values)


class CountingBackend(FakeBackend):
    def __init__(self):
        super().__init__(device='cpu')
        self.calls = []

    def prefill(self, tokens, past_key_values=None):
        self.calls.append((tuple(tokens), past_key_values is not None))
        return super().prefill(tokens, past_key_values=past_key_values)


def test_reactive_miss_accounts_for_prefix_storage_latency():
    backend = CountingBackend()
    engine = ReactivePrefixCacheEngine(backend=backend)
    tokens = (1, 2, 3, 4, 5, 6)

    result = engine.serve_tokens(1, tokens)

    assert len(backend.calls) == 2
    full_tokens, used_kv_a = backend.calls[0]
    prefix_tokens, used_kv_b = backend.calls[1]
    assert full_tokens == tokens
    assert prefix_tokens == tokens
    assert not used_kv_a and not used_kv_b
    expected = (2.5 * len(full_tokens) + 0.25) + (2.5 * len(prefix_tokens) + 0.25)
    assert result.latency_ms == expected


def test_reactive_engine_falls_back_to_full_recompute_when_cached_prefill_fails():
    backend = FailingCacheBackend()
    engine = ReactivePrefixCacheEngine(backend=backend)
    cached_tokens = (1, 2, 3, 4, 5, 6)
    extended_tokens = (1, 2, 3, 4, 5, 6, 7, 8)

    engine.serve_tokens(1, cached_tokens)
    result = engine.serve_tokens(2, extended_tokens)

    assert result.was_cache_hit is False
    assert result.tokens_recomputed == len(extended_tokens)


def test_shadowkv_reports_promoted_gpu_tier_consistently():
    backend = FakeBackend(device='cuda:0')
    engine = ShadowKVEngine(backend=backend, enable_gpu_tier=True)
    engine.stop_event.set()
    engine.thread.join(timeout=1.0)
    tokens = (1, 2, 3, 4, 5, 6)

    engine.bank.observe_query(tokens)
    engine.bank.store(tokens[:3], {'prefix_len': 3, 'device': 'cpu'}, 1.0, 128, is_speculative=True, tier='cpu')
    entry = engine.bank.entries[tokens[:3]]
    entry.hit_count = 2

    result = engine.serve_tokens(2, tokens)

    assert result.was_cache_hit is True
    assert result.cache_tier == 'gpu'
