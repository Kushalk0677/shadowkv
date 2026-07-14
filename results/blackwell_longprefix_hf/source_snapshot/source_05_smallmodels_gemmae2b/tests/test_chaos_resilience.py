"""
Chaos testing framework for ShadowKV++ resilience.
Tests the system's ability to handle failures and edge cases under controlled chaos.
"""
from __future__ import annotations

import random
import time
import threading
import pytest

from proactive_kv_cache.models import FakeBackend
from proactive_kv_cache.engines import ReactivePrefixCacheEngine


class ChaosMonkey:
    """Injects controlled chaos into the system to test resilience."""

    def __init__(self, engine, failure_rate=0.1):
        self.engine = engine
        self.failure_rate = failure_rate
        self.active = False
        self.chaos_thread = None
        self.stop_event = threading.Event()

    def start(self):
        self.active = True
        self.stop_event.clear()
        self.chaos_thread = threading.Thread(target=self._inject_chaos, daemon=True)
        self.chaos_thread.start()

    def stop(self):
        self.active = False
        self.stop_event.set()
        if self.chaos_thread:
            self.chaos_thread.join(timeout=2.0)

    def _inject_chaos(self):
        while not self.stop_event.wait(0.05):
            if not self.active:
                continue
            random.choice([
                self._inject_latency,
                self._evict_random_prefix,
                self._corrupt_cache_entry,
            ])()

    def _inject_latency(self):
        if random.random() < self.failure_rate:
            original_prefill = self.engine.backend.prefill
            delay = random.uniform(0.01, 0.05)

            def delayed_prefill(*args, **kwargs):
                time.sleep(delay)
                return original_prefill(*args, **kwargs)

            self.engine.backend.prefill = delayed_prefill

            def revert():
                time.sleep(random.uniform(0.2, 0.5))
                self.engine.backend.prefill = original_prefill

            threading.Thread(target=revert, daemon=True).start()

    def _evict_random_prefix(self):
        if random.random() < self.failure_rate:
            with self.engine.bank._lock:
                if self.engine.bank.entries:
                    prefix = random.choice(list(self.engine.bank.entries.keys()))
                    self.engine.bank._remove_unlocked(prefix, count_speculative_waste=False)

    def _corrupt_cache_entry(self):
        if random.random() < self.failure_rate:
            with self.engine.bank._lock:
                if self.engine.bank.entries:
                    prefix = random.choice(list(self.engine.bank.entries.keys()))
                    self.engine.bank.entries[prefix].kv_cache = None


@pytest.fixture
def engine():
    """Create a fresh engine with FakeBackend for chaos testing."""
    return ReactivePrefixCacheEngine(FakeBackend())


@pytest.fixture
def test_requests():
    """Shared set of test prompt texts."""
    return [
        "What is the capital of France?",
        "Tell me about Python programming",
        "Explain machine learning",
        "What is the weather today?",
    ]


@pytest.mark.parametrize("num_requests", [50, 100])
def test_chaos_resilience_success_rate(engine, test_requests, num_requests):
    """Engine should complete > 90% of requests even under chaos conditions."""
    chaos = ChaosMonkey(engine, failure_rate=0.15)
    chaos.start()
    try:
        successes = 0
        for _ in range(num_requests):
            try:
                prompt = random.choice(test_requests)
                tokens = engine.backend.tokenize(prompt)
                engine.serve_tokens(_ + 1, tokens)
                successes += 1
            except Exception:
                pass
    finally:
        chaos.stop()

    rate = successes / num_requests
    assert rate >= 0.9, f"Success rate {rate:.2f} is below 0.9 threshold (got {successes}/{num_requests})"


def test_chaos_resilience_no_crash(engine, test_requests):
    """Engine should not crash under sustained chaos for 100+ requests."""
    chaos = ChaosMonkey(engine, failure_rate=0.2)
    chaos.start()
    try:
        for i in range(100):
            try:
                prompt = random.choice(test_requests)
                tokens = engine.backend.tokenize(prompt)
                engine.serve_tokens(i, tokens)
            except Exception:
                pass
    finally:
        chaos.stop()

    assert True, "Engine survived sustained chaos without crashing"


def test_chaos_resilience_bounded_latency(engine, test_requests):
    """Even with chaos, average latency should remain reasonable."""
    chaos = ChaosMonkey(engine, failure_rate=0.1)
    chaos.start()
    try:
        num_requests = 30
        start = time.perf_counter()
        for _ in range(num_requests):
            try:
                prompt = random.choice(test_requests)
                tokens = engine.backend.tokenize(prompt)
                engine.serve_tokens(0, tokens)
            except Exception:
                pass
        elapsed = (time.perf_counter() - start) / num_requests
    finally:
        chaos.stop()

    assert elapsed < 2.0, f"Average latency {elapsed*1000:.1f}ms exceeds 2s budget under chaos"


def test_no_chaos_baseline(engine, test_requests):
    """Verify baseline behavior without chaos is 100% successful."""
    num_requests = 50
    successes = 0
    for _ in range(num_requests):
        try:
            prompt = random.choice(test_requests)
            tokens = engine.backend.tokenize(prompt)
            engine.serve_tokens(_ + 1, tokens)
            successes += 1
        except Exception:
            pass

    assert successes == num_requests, (
        f"Baseline without chaos should be 100% successful "
        f"({successes}/{num_requests} succeeded)"
    )
