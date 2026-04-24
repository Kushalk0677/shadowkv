from proactive_kv_cache.cache import TieredStateBank


def test_store_and_lookup_longest_prefix():
    bank = TieredStateBank(max_memory_bytes=4096, min_match_length=3)
    bank.observe_query((1,2,3,4,5,6))
    bank.store((1,2,3), {'x': 1}, generation_cost_ms=10.0, memory_bytes=100, is_speculative=True, tier='cpu')
    bank.store((1,2,3,4), {'x': 2}, generation_cost_ms=11.0, memory_bytes=100, is_speculative=False, tier='cpu')
    match = bank.lookup((1,2,3,4,9))
    assert match is not None
    key, entry, match_len = match
    assert key == (1,2,3,4)
    assert match_len == 4
    assert entry.was_speculative is False


def test_eviction_under_budget():
    bank = TieredStateBank(max_memory_bytes=150, min_match_length=3)
    bank.observe_query((1,2,3,4))
    assert bank.store((1,2,3), {'a': 1}, generation_cost_ms=10.0, memory_bytes=100, is_speculative=True, tier='cpu')
    assert bank.store((1,2,3,4), {'b': 2}, generation_cost_ms=20.0, memory_bytes=100, is_speculative=False, tier='cpu')
    assert bank.current_memory_bytes <= 150
    assert bank.snapshot_metrics()['evictions'] >= 1
