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


def test_default_prefix_lengths_tracks_dense_short_and_stride_long_prefixes():
    bank = TieredStateBank(max_memory_bytes=4096, min_match_length=3)
    lengths = bank.default_prefix_lengths(tuple(range(1, 65)))
    assert 3 in lengths
    assert 16 in lengths
    assert 24 in lengths
    assert 32 in lengths
    assert 40 in lengths
    assert 48 in lengths
    assert 64 in lengths


def test_observe_query_respects_reusable_prefix_limit_and_tracks_branching():
    bank = TieredStateBank(max_memory_bytes=4096, min_match_length=3)
    tokens_a = (1, 2, 3, 4, 5, 6, 7, 8)
    tokens_b = (1, 2, 3, 4, 9, 10, 11, 12)
    bank.observe_query(tokens_a, tracked_prefix_lengths=[3, 4, 5, 6], reusable_prefix_limit=4, observed_at=10.0)
    bank.observe_query(tokens_b, tracked_prefix_lengths=[3, 4, 5, 6], reusable_prefix_limit=4, observed_at=11.0)

    assert bank.get_observation_count((1, 2, 3, 4)) == 2
    assert bank.get_observation_count((1, 2, 3, 4, 5)) == 0
    assert bank.branching_factor((1, 2, 3, 4)) == 2
    assert bank.recent_arrival_rate() > 0.0
