from proactive_kv_cache.datasets import _row_to_prompt
from proactive_kv_cache.engines import ReactivePrefixCacheEngine
from proactive_kv_cache.models import FakeBackend


class ConservativeNoReuseBackend(FakeBackend):
    def __init__(self):
        super().__init__(device='cpu')
        self.default_min_store_prefix_tokens = 3
        self.default_min_reuse_prefix_tokens = 3


def test_summarization_prompt_does_not_leak_reference_summary():
    prompt = _row_to_prompt('summarization', {'document': 'Doc body', 'summary': 'gold summary'})
    assert 'Reference summary:' not in prompt
    assert 'gold summary' not in prompt


def test_instruction_prompt_does_not_leak_reference_answer():
    prompt = _row_to_prompt('instruction', {'instruction': 'Do thing', 'context': 'ctx', 'response': 'gold answer'})
    assert 'Reference answer:' not in prompt
    assert 'gold answer' not in prompt


def test_reactive_engine_auto_disables_after_repeated_unusable_matches():
    backend = ConservativeNoReuseBackend()
    engine = ReactivePrefixCacheEngine(backend=backend)
    engine.tuning.min_estimated_saved_ms = 10_000.0  # make every discovered match too expensive to use
    requests = [
        (1, 2, 3, 4),
        (1, 2, 3, 5),
        (1, 2, 3, 6),
        (1, 2, 3, 7),
        (1, 2, 3, 8),
        (1, 2, 3, 9),
        (1, 2, 3, 10),
        (1, 2, 3, 11),
        (1, 2, 3, 12),
        (1, 2, 3, 13),
        (1, 2, 3, 14),
        (1, 2, 3, 15),
    ]
    for i, tokens in enumerate(requests):
        engine.serve_tokens(i, tokens)
    assert engine.cache_enabled is False
    assert engine.engine_metrics['cache_active_final'] is False
    assert engine.engine_metrics['auto_disabled_reason'] in {'no_usable_reuse', 'net_negative_reuse', 'low_reuse_density', 'no_prefix_reuse_early'}


def test_reactive_engine_keeps_cache_enabled_when_absolute_reuse_is_meaningful_even_if_density_is_low():
    backend = ConservativeNoReuseBackend()
    engine = ReactivePrefixCacheEngine(backend=backend)
    engine.engine_metrics['requests_seen'] = 16
    engine.engine_metrics['reuse_attempts'] = 16
    engine.engine_metrics['reuse_successes'] = 1
    engine.engine_metrics['bypassed_matches'] = 15
    engine.engine_metrics['recompute_tokens_total'] = 6400
    engine.engine_metrics['reused_prefix_tokens_total'] = 96

    engine._maybe_auto_disable()

    assert engine.cache_enabled is True
    assert engine.engine_metrics['auto_disabled_reason'] is None
