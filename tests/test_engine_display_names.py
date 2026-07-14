from proactive_kv_cache.engine_names import display_engine_name, display_engine_names


def test_engine_display_names_keep_raw_ids_stable():
    assert display_engine_name('shadow_kv') == 'MeritKV-Sem'
    assert display_engine_name('shadow_kv_plus') == 'MeritKV'
    assert display_engine_name('shadow_kv_plus_lite') == 'MeritKV-Lite'
    assert display_engine_name('shadow_kv_plus_best_latency') == 'MeritKV-BestLatency'
    assert display_engine_name('shadow_kv_plus_raw_observer') == 'MeritKV-RawObserver'


def test_engine_display_names_preserve_unknown_ids():
    assert display_engine_name('no_cache') == 'no_cache'
    assert display_engine_names(('shadow_kv', 'no_cache')) == {
        'shadow_kv': 'MeritKV-Sem',
        'no_cache': 'no_cache',
    }