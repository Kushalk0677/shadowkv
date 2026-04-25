import time
import threading

from proactive_kv_cache.cache import TieredStateBank
from proactive_kv_cache.engines import FrequencySpeculativeEngine, ReactivePrefixCacheEngine, ShadowKVEngine
from proactive_kv_cache.models import FakeBackend, PrefillResult
from proactive_kv_cache.policy import CostAwareSlackPolicy, FrequencyPolicy


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


class InternalFallbackBackend(FakeBackend):
    def prefill(self, tokens, past_key_values=None):
        if past_key_values is not None:
            result = super().prefill(tokens, past_key_values=None)
            return PrefillResult(
                kv_cache=result.kv_cache,
                latency_ms=result.latency_ms,
                memory_bytes=result.memory_bytes,
                device=result.device,
                gpu_utilization_pct=result.gpu_utilization_pct,
                used_past_key_values=False,
                cache_fallback_reason='backend_retry_without_cache:RuntimeError',
                prepared_past_length=0,
            )
        return super().prefill(tokens, past_key_values=past_key_values)


class SlowSpeculationBackend(FakeBackend):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device=device)
        self.call_count = 0
        self.speculation_started = threading.Event()

    def prefill(self, tokens, past_key_values=None):
        self.call_count += 1
        if self.call_count == 2 and past_key_values is None:
            self.speculation_started.set()
            time.sleep(0.05)
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


def test_reactive_engine_treats_internal_backend_cache_fallback_as_miss():
    backend = InternalFallbackBackend()
    engine = ReactivePrefixCacheEngine(backend=backend)
    cached_tokens = (1, 2, 3, 4, 5, 6)
    extended_tokens = (1, 2, 3, 4, 5, 6, 7)

    engine.serve_tokens(1, cached_tokens)
    result = engine.serve_tokens(2, extended_tokens)

    assert result.was_cache_hit is False
    assert result.tokens_recomputed == len(extended_tokens)
    assert engine.engine_metrics['reuse_failures'] >= 1


def test_shadowkv_reports_promoted_gpu_tier_consistently():
    backend = FakeBackend(device='cuda:0')
    engine = ShadowKVEngine(backend=backend, enable_gpu_tier=True)
    engine.stop_event.set()
    if engine.thread is not None:
        engine.thread.join(timeout=1.0)
    tokens = (1, 2, 3, 4, 5, 6)

    engine.bank.observe_query(tokens)
    engine.bank.store(tokens[:3], {'prefix_len': 3, 'device': 'cpu'}, 1.0, 128, is_speculative=True, tier='cpu')
    entry = engine.bank.entries[tokens[:3]]
    entry.hit_count = 2

    result = engine.serve_tokens(2, tokens)

    assert result.was_cache_hit is True
    assert result.cache_tier == 'gpu'


def test_shadowkv_controller_pauses_when_pending_speculation_is_already_outstanding():
    backend = FakeBackend(device='cpu')
    engine = ShadowKVEngine(backend=backend, enable_gpu_tier=False, speculative_k=2, idle_threshold_ms=1.0)
    engine.stop_event.set()
    if engine.thread is not None:
        engine.thread.join(timeout=1.0)

    engine.engine_metrics['requests_seen'] = 20
    first = (1, 2, 3, 4)
    second = (5, 6, 7, 8)
    engine.bank.store(first, {'prefix_len': 4, 'device': 'cpu'}, 5.0, 128, is_speculative=True, tier='cpu')
    engine.bank.store(second, {'prefix_len': 4, 'device': 'cpu'}, 5.0, 128, is_speculative=True, tier='cpu')

    allowed = engine._refresh_speculation_controller()

    assert allowed is False
    assert engine.engine_metrics['effective_speculative_k_final'] == 0


def test_shadowkv_controller_enters_cooldown_after_wasted_speculation():
    backend = FakeBackend(device='cpu')
    engine = ShadowKVEngine(backend=backend, enable_gpu_tier=False, speculative_k=2, idle_threshold_ms=20.0)
    engine.stop_event.set()
    if engine.thread is not None:
        engine.thread.join(timeout=1.0)

    engine.engine_metrics['requests_seen'] = 20
    prefix = (1, 2, 3, 4)
    engine.bank.store(prefix, {'prefix_len': 4, 'device': 'cpu'}, 5.0, 128, is_speculative=True, tier='cpu')
    engine.bank.remove(prefix)

    allowed = engine._refresh_speculation_controller()

    assert allowed is False
    assert engine.engine_metrics['speculation_cooldown_events'] >= 1


def test_shadowkv_controller_pauses_when_recent_reuse_density_is_too_low():
    backend = FakeBackend(device='cpu')
    engine = ShadowKVEngine(backend=backend, enable_gpu_tier=False, speculative_k=2, idle_threshold_ms=1.0)
    engine.stop_event.set()
    if engine.thread is not None:
        engine.thread.join(timeout=1.0)

    engine.engine_metrics['requests_seen'] = 20
    engine._recent_request_window.extend(
        [
            (64, 0, False),
            (60, 0, False),
            (72, 0, False),
            (58, 0, False),
            (70, 0, False),
            (66, 0, False),
        ]
    )

    allowed = engine._refresh_speculation_controller()

    assert allowed is False
    assert engine.engine_metrics['recent_reuse_density_final'] == 0.0


def test_shadowkv_cpu_controller_opens_up_when_recent_reuse_is_strong():
    backend = FakeBackend(device='cpu')
    engine = ShadowKVEngine(backend=backend, enable_gpu_tier=False, speculative_k=2, idle_threshold_ms=1.0)
    engine.stop_event.set()
    if engine.thread is not None:
        engine.thread.join(timeout=1.0)

    engine.engine_metrics['requests_seen'] = 20
    engine._recent_request_window.extend(
        [
            (64, 24, True),
            (60, 20, True),
            (72, 24, True),
            (58, 18, True),
            (70, 20, True),
            (66, 22, True),
        ]
    )
    engine._recent_speculative_net_values.append(18.0)

    allowed = engine._refresh_speculation_controller()

    assert allowed is True
    assert engine.engine_metrics['effective_speculative_k_final'] == 2


def test_shadowkv_controller_keeps_speculation_alive_when_absolute_reuse_is_meaningful_but_density_is_low():
    backend = FakeBackend(device='cpu')
    engine = ShadowKVEngine(backend=backend, enable_gpu_tier=False, speculative_k=2, idle_threshold_ms=1.0)
    engine.stop_event.set()
    if engine.thread is not None:
        engine.thread.join(timeout=1.0)

    engine.engine_metrics['requests_seen'] = 20
    engine._recent_request_window.extend(
        [
            (1200, 94, True),
            (1100, 0, False),
            (1150, 0, False),
            (1180, 0, False),
            (1120, 0, False),
            (1210, 0, False),
            (1170, 0, False),
            (1110, 0, False),
            (1190, 0, False),
            (1160, 0, False),
            (1130, 0, False),
            (1140, 0, False),
        ]
    )
    engine._recent_speculative_net_values.append(8.0)

    allowed = engine._refresh_speculation_controller()

    assert allowed is True
    assert engine.engine_metrics['effective_speculative_k_final'] >= 1


def test_shadowkv_bootstrap_defers_reactive_store_and_allows_speculative_hit():
    backend = FakeBackend(device='cpu')
    policy = CostAwareSlackPolicy(
        min_frequency=0.0,
        benefit_cost_ratio=0.0,
        fixed_prefill_overhead_ms=0.1,
        memory_penalty_per_mb=0.0,
        max_admissions_per_idle=1,
        min_prefix_len=3,
        max_prefix_len=24,
        min_observations=1,
        min_expected_net_ms=0.0,
        min_recent_support=0.0,
        min_utility_score=0.0,
    )
    engine = ShadowKVEngine(backend=backend, enable_gpu_tier=False, speculative_k=1, idle_threshold_ms=1.0, policy=policy)
    tokens = (1, 2, 3, 4, 5, 6)
    metadata = {'prompt_mode': 'templated', 'shared_prefix_text': 'shared prompt', 'shared_prefix_hint_tokens': 4}

    first = engine.serve_tokens(1, tokens, metadata=metadata)
    assert first.was_cache_hit is False
    assert engine.engine_metrics['store_successes'] == 0
    assert engine.engine_metrics['bootstrap_store_deferrals'] >= 1

    time.sleep(0.05)

    second = engine.serve_tokens(2, tokens, metadata=metadata)
    engine.shutdown()

    assert second.was_cache_hit is True
    assert second.was_speculative_hit is True
    assert len(engine.speculative_log) >= 1
    assert engine.engine_metrics['speculative_useful_savings_ms'] > 0.0


def test_cost_aware_policy_uses_bootstrap_horizon_without_arrival_history():
    bank = TieredStateBank(max_memory_bytes=32 * 1024 * 1024)
    tokens = (1, 2, 3, 4, 5, 6)
    bank.observe_query(tokens, tracked_prefix_lengths=[4], reusable_prefix_limit=4, observed_at=1.0)

    policy = CostAwareSlackPolicy(
        min_frequency=0.0,
        ms_per_token=0.4,
        fixed_prefill_overhead_ms=0.1,
        memory_penalty_per_mb=0.0,
        kv_mb_per_token=0.001,
        idle_cost_fraction=0.1,
        benefit_cost_ratio=0.0,
        max_admissions_per_idle=1,
        min_prefix_len=4,
        max_prefix_len=8,
        min_observations=1,
        min_expected_net_ms=0.0,
        min_recent_support=0.0,
        min_utility_score=0.0,
        bootstrap_horizon_requests=1.0,
    )

    decisions = policy.rank(bank, budget_k=1, prefer_gpu=False)

    assert len(decisions) == 1
    assert decisions[0].expected_benefit_ms > decisions[0].expected_cost_ms


def test_frequency_speculative_bootstrap_defers_reactive_store_and_records_speculation():
    backend = FakeBackend(device='cpu')
    engine = FrequencySpeculativeEngine(backend=backend, speculative_k=1, idle_threshold_ms=1.0)
    engine.policy = FrequencyPolicy(min_frequency=0.0, min_prefix_len=3, max_prefix_len=24, min_observations=1)
    engine._bootstrap_speculation_observations = 1
    tokens = (1, 2, 3, 4, 5, 6)
    metadata = {'prompt_mode': 'templated', 'shared_prefix_text': 'shared prompt', 'shared_prefix_hint_tokens': 4}

    first = engine.serve_tokens(1, tokens, metadata=metadata)
    assert first.was_cache_hit is False
    assert engine.engine_metrics['bootstrap_store_deferrals'] >= 1

    time.sleep(0.05)

    second = engine.serve_tokens(2, tokens, metadata=metadata)
    engine.shutdown()

    assert second.was_cache_hit is True
    assert second.was_speculative_hit is True
    assert len(engine.speculative_log) >= 1


def test_frequency_speculative_records_overlap_when_request_arrives_during_speculation():
    backend = SlowSpeculationBackend(device='cpu')
    engine = FrequencySpeculativeEngine(backend=backend, speculative_k=1, idle_threshold_ms=1.0)
    engine.policy = FrequencyPolicy(min_frequency=0.0, min_prefix_len=3, max_prefix_len=24, min_observations=1)
    engine._bootstrap_speculation_observations = 1
    tokens = (1, 2, 3, 4, 5, 6)
    metadata = {'prompt_mode': 'templated', 'shared_prefix_text': 'shared prompt', 'shared_prefix_hint_tokens': 4}

    engine.serve_tokens(1, tokens, metadata=metadata)
    assert backend.speculation_started.wait(timeout=1.0)
    engine.serve_tokens(2, tokens, metadata=metadata)
    engine.shutdown()

    assert engine.engine_metrics['speculative_overlap_events'] >= 1
    assert engine.engine_metrics['speculative_overlap_ms'] > 0.0


def test_shadowkv_promotes_hot_speculative_entry_to_gpu():
    backend = FakeBackend(device='cuda:0')
    policy = CostAwareSlackPolicy(
        min_frequency=0.0,
        benefit_cost_ratio=0.0,
        fixed_prefill_overhead_ms=0.1,
        memory_penalty_per_mb=0.0,
        max_admissions_per_idle=1,
        min_prefix_len=3,
        max_prefix_len=32,
        min_observations=1,
        min_expected_net_ms=0.0,
        min_recent_support=0.0,
        min_utility_score=0.0,
        bootstrap_horizon_requests=1.0,
    )
    engine = ShadowKVEngine(backend=backend, enable_gpu_tier=True, speculative_k=1, idle_threshold_ms=1.0, policy=policy)
    tokens = tuple(range(1, 17))
    metadata = {'prompt_mode': 'templated', 'shared_prefix_text': 'shared prompt', 'shared_prefix_hint_tokens': 12}

    first = engine.serve_tokens(1, tokens, metadata=metadata)
    assert first.was_cache_hit is False
    time.sleep(0.05)
    second = engine.serve_tokens(2, tokens, metadata=metadata)
    engine.shutdown()

    assert second.was_cache_hit is True
    assert second.cache_tier == 'gpu'
    assert engine.bank.snapshot_metrics()['promotions'] >= 1
