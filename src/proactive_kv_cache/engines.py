from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .cache import CacheEntry, TieredStateBank
from .metrics import summarize_run
from .models import Backend
from .policy import CostAwareSlackPolicy, FrequencyPolicy, SpeculationPolicy


@dataclass
class RequestResult:
    request_id: int
    latency_ms: float
    matched_prefix_length: int
    tokens_recomputed: int
    was_cache_hit: bool
    was_speculative_hit: bool
    cache_tier: str | None = None
    gpu_utilization_pct: float | None = None


@dataclass
class EngineTuning:
    min_store_prefix_tokens: int
    min_reuse_prefix_tokens: int
    min_estimated_saved_ms: float
    max_cacheable_prefix_tokens: int
    min_prefix_coverage_ratio: float
    promote_after_hits: int
    gpu_promotion_max_prefix_tokens: int


class BaseEngine:
    def __init__(self, backend: Backend, max_memory_mb: int = 256, name: str = 'base'):
        self.backend = backend
        self.name = name
        self.bank = TieredStateBank(max_memory_bytes=max_memory_mb * 1024 ** 2)
        self.latencies: List[float] = []
        self.results: List[RequestResult] = []
        self.gpu_utils: List[float | None] = []
        self.start_time = time.perf_counter()
        self.end_time = self.start_time
        self.engine_metrics: Dict[str, float | int | bool | None] = {
            'requests_seen': 0,
            'cache_active_final': True,
            'auto_disabled_reason': None,
            'reuse_attempts': 0,
            'reuse_successes': 0,
            'reuse_failures': 0,
            'store_attempts': 0,
            'store_successes': 0,
            'store_skips': 0,
            'bypassed_matches': 0,
            'reused_prefix_tokens_total': 0,
            'recompute_tokens_total': 0,
            'estimated_tokens_saved_total': 0,
            'saved_latency_estimate_ms': 0.0,
            'store_latency_total_ms': 0.0,
            'full_prefill_latency_total_ms': 0.0,
            'cache_disable_check_every': 8,
            'cache_disable_min_attempts': 8,
            'cache_disable_min_requests': 10,
        }
        self.cache_enabled = True
        self.ewma_full_ms_per_token = 0.0
        self.ewma_reuse_ms_per_token = 0.0
        self.ewma_cache_reuse_overhead_ms = float(getattr(backend, 'default_cache_reuse_overhead_ms', 2.0))
        self._stats_alpha = 0.20
        self.tuning = self._default_tuning()

    def _default_tuning(self) -> EngineTuning:
        if getattr(self.backend, 'backend_name', 'base') == 'hf' and not self.backend.device.startswith('cuda'):
            return EngineTuning(
                min_store_prefix_tokens=max(getattr(self.backend, 'default_min_store_prefix_tokens', 12), 12),
                min_reuse_prefix_tokens=max(getattr(self.backend, 'default_min_reuse_prefix_tokens', 16), 16),
                min_estimated_saved_ms=12.0,
                max_cacheable_prefix_tokens=96,
                min_prefix_coverage_ratio=0.28,
                promote_after_hits=3,
                gpu_promotion_max_prefix_tokens=24,
            )
        if getattr(self.backend, 'backend_name', 'base') == 'hf' and self.backend.device.startswith('cuda'):
            return EngineTuning(
                min_store_prefix_tokens=max(getattr(self.backend, 'default_min_store_prefix_tokens', 8), 8),
                min_reuse_prefix_tokens=max(getattr(self.backend, 'default_min_reuse_prefix_tokens', 8), 8),
                min_estimated_saved_ms=4.0,
                max_cacheable_prefix_tokens=128,
                min_prefix_coverage_ratio=0.15,
                promote_after_hits=2,
                gpu_promotion_max_prefix_tokens=64,
            )
        return EngineTuning(
            min_store_prefix_tokens=max(getattr(self.backend, 'default_min_store_prefix_tokens', 8), 3),
            min_reuse_prefix_tokens=max(getattr(self.backend, 'default_min_reuse_prefix_tokens', 8), 3),
            min_estimated_saved_ms=1.0 if getattr(self.backend, 'backend_name', 'base') == 'fake' else 6.0,
            max_cacheable_prefix_tokens=96,
            min_prefix_coverage_ratio=0.20,
            promote_after_hits=3,
            gpu_promotion_max_prefix_tokens=24,
        )


    def _set_cache_disabled(self, reason: str) -> None:
        if self.engine_metrics.get('auto_disabled_reason') is None:
            self.engine_metrics['auto_disabled_reason'] = reason
        self.cache_enabled = False
        self.engine_metrics['cache_active_final'] = False

    def _mark_cache_active(self) -> None:
        self.engine_metrics['cache_active_final'] = bool(self.cache_enabled)

    def _after_record(self, result: RequestResult) -> None:
        return None

    def _on_request_finish(self) -> None:
        self.engine_metrics['requests_seen'] = int(self.engine_metrics.get('requests_seen', 0)) + 1
        self._maybe_auto_disable()
        self._mark_cache_active()

    def _maybe_auto_disable(self) -> None:
        if not self.cache_enabled:
            return
        if self.name in ('no_cache', 'native_prefix_cache'):
            return
        requests_seen = int(self.engine_metrics.get('requests_seen', 0))
        attempts = int(self.engine_metrics.get('reuse_attempts', 0))
        successes = int(self.engine_metrics.get('reuse_successes', 0))
        bypassed = int(self.engine_metrics.get('bypassed_matches', 0))
        store_successes = int(self.engine_metrics.get('store_successes', 0))
        min_requests = int(self.engine_metrics.get('cache_disable_min_requests', 10))
        min_attempts = int(self.engine_metrics.get('cache_disable_min_attempts', 8))
        bank_snapshot = self.bank.snapshot_metrics()
        if requests_seen >= 5:
            all_misses = int(bank_snapshot.get('hits', 0)) == 0 and int(bank_snapshot.get('misses', 0)) >= requests_seen
            if all_misses and attempts == 0 and store_successes >= 4:
                self._set_cache_disabled('no_prefix_reuse_early')
                return
        if requests_seen < min_requests or attempts < min_attempts:
            return
        success_rate = successes / max(attempts, 1)
        bypass_rate = bypassed / max(attempts, 1)
        saved_ms = float(self.engine_metrics.get('saved_latency_estimate_ms', 0.0))
        store_ms = float(self.engine_metrics.get('store_latency_total_ms', 0.0))
        # We tried a few times and still could not reuse anything useful.
        if successes == 0 and bypass_rate >= 0.80:
            self._set_cache_disabled('no_usable_reuse')
            return
        # Storing prefixes is costing more than it seems to save.
        if success_rate < 0.05 and saved_ms < (store_ms * 0.75):
            self._set_cache_disabled('net_negative_reuse')
            return
        # The workload does not have enough shared prefix material.
        recompute = int(self.engine_metrics.get('recompute_tokens_total', 0))
        reused = int(self.engine_metrics.get('reused_prefix_tokens_total', 0))
        if recompute > 0 and reused / max(recompute + reused, 1) < 0.03 and bypass_rate >= 0.75:
            self._set_cache_disabled('low_reuse_density')
            return

    def finalize(self) -> None:
        self._mark_cache_active()
        self.end_time = time.perf_counter()

    def _record(self, result: RequestResult) -> RequestResult:
        self.latencies.append(result.latency_ms)
        self.results.append(result)
        self.gpu_utils.append(result.gpu_utilization_pct)
        self.engine_metrics['recompute_tokens_total'] = int(self.engine_metrics.get('recompute_tokens_total', 0)) + int(result.tokens_recomputed)
        if result.was_cache_hit and result.matched_prefix_length > 0:
            self.engine_metrics['reused_prefix_tokens_total'] = int(self.engine_metrics.get('reused_prefix_tokens_total', 0)) + int(result.matched_prefix_length)
        self._after_record(result)
        self._on_request_finish()
        self.end_time = time.perf_counter()
        return result

    def _device_target_for_tier(self, tier: str) -> str:
        if tier == 'gpu':
            return self.backend.device
        return tier

    def _update_full_cost_stats(self, token_count: int, latency_ms: float) -> None:
        if token_count <= 0:
            return
        observed = latency_ms / max(token_count, 1)
        if self.ewma_full_ms_per_token <= 0.0:
            self.ewma_full_ms_per_token = observed
        else:
            self.ewma_full_ms_per_token = (1.0 - self._stats_alpha) * self.ewma_full_ms_per_token + self._stats_alpha * observed

    def _update_reuse_cost_stats(self, suffix_token_count: int, latency_ms: float, matched_prefix_length: int) -> None:
        if suffix_token_count > 0:
            observed = latency_ms / suffix_token_count
            if self.ewma_reuse_ms_per_token <= 0.0:
                self.ewma_reuse_ms_per_token = observed
            else:
                self.ewma_reuse_ms_per_token = (1.0 - self._stats_alpha) * self.ewma_reuse_ms_per_token + self._stats_alpha * observed
        full_ms_per_token = self.ewma_full_ms_per_token or max(self.backend.estimate_prefill_cost_ms(max(suffix_token_count, 1)) / max(suffix_token_count, 1), 1e-6)
        inferred_overhead = max(latency_ms - full_ms_per_token * max(suffix_token_count, 0), 0.0)
        if matched_prefix_length > 0 or suffix_token_count == 0:
            self.ewma_cache_reuse_overhead_ms = (
                (1.0 - self._stats_alpha) * self.ewma_cache_reuse_overhead_ms + self._stats_alpha * inferred_overhead
            )

    def _estimate_full_cost_ms(self, token_count: int) -> float:
        if token_count <= 0:
            return 0.0
        if self.ewma_full_ms_per_token > 0.0:
            return self.ewma_full_ms_per_token * token_count
        return self.backend.estimate_prefill_cost_ms(token_count)

    def _estimate_saved_ms(self, total_tokens: int, match_len: int, suffix_len: int) -> float:
        matched_cost = self._estimate_full_cost_ms(match_len)
        suffix_cost = self._estimate_full_cost_ms(suffix_len)
        gross_saved = max(matched_cost, 0.0)
        penalty = self.ewma_cache_reuse_overhead_ms + 0.05 * max(suffix_len - match_len, 0)
        if total_tokens > 0 and match_len / total_tokens < self.tuning.min_prefix_coverage_ratio:
            penalty += 4.0
        return gross_saved - penalty - 0.10 * suffix_cost

    def _reactive_prefix_len(self, tokens: Tuple[int, ...]) -> int:
        candidate_lengths = self.bank.default_prefix_lengths(tokens)
        best_len = min(len(tokens), max(self.bank.min_match_length, self.tuning.min_store_prefix_tokens))
        best_score = float('-inf')

        for length in candidate_lengths:
            if (
                length < self.bank.min_match_length
                or length > len(tokens)
                or length < self.tuning.min_store_prefix_tokens
                or length > self.tuning.max_cacheable_prefix_tokens
            ):
                continue
            prefix = tokens[:length]
            freq = self.bank.get_frequency(prefix)
            coverage = length / max(len(tokens), 1)
            score = (freq * (1.0 + 0.05 * min(length, 24))) + (0.5 * coverage) - (0.015 * max(length - 32, 0))
            if score > best_score:
                best_score = score
                best_len = length

        return best_len

    def _should_store_reactive_prefix(self, tokens: Tuple[int, ...], prefix_len: int) -> bool:
        self.engine_metrics['store_attempts'] += 1
        if not getattr(self.backend, 'supports_external_kv', True):
            self.engine_metrics['store_skips'] += 1
            return False
        if len(tokens) < self.tuning.min_store_prefix_tokens:
            self.engine_metrics['store_skips'] += 1
            return False
        if prefix_len < self.tuning.min_store_prefix_tokens or prefix_len > self.tuning.max_cacheable_prefix_tokens:
            self.engine_metrics['store_skips'] += 1
            return False
        freq = self.bank.get_frequency(tokens[:prefix_len])
        observations = self.bank.get_observation_count(tokens[:prefix_len])
        est_cost = self._estimate_full_cost_ms(prefix_len)
        if observations < 2 and est_cost < self.tuning.min_estimated_saved_ms * 1.5:
            self.engine_metrics['store_skips'] += 1
            return False
        if freq < 0.08 and est_cost < self.tuning.min_estimated_saved_ms:
            self.engine_metrics['store_skips'] += 1
            return False
        return True

    def _store_reactive_prefix(self, tokens: Tuple[int, ...], tier: str = 'cpu') -> float:
        prefix_len = self._reactive_prefix_len(tokens)
        if not self._should_store_reactive_prefix(tokens, prefix_len):
            return 0.0
        prefix = tokens[:prefix_len]
        prefill = self.backend.prefill(prefix)
        self._update_full_cost_stats(prefix_len, prefill.latency_ms)
        device_target = self._device_target_for_tier(tier)
        cache_obj = prefill.kv_cache if device_target == self.backend.device else self.backend.move_kv_cache(prefill.kv_cache, device_target)
        stored = self.bank.store(prefix, cache_obj, prefill.latency_ms, prefill.memory_bytes, is_speculative=False, tier=tier)
        if stored:
            self.engine_metrics['store_successes'] += 1
            self.engine_metrics['store_latency_total_ms'] = float(self.engine_metrics.get('store_latency_total_ms', 0.0)) + float(prefill.latency_ms)
            return float(prefill.latency_ms)
        self.engine_metrics['store_skips'] += 1
        return 0.0

    def _should_attempt_cache_use(self, tokens: Tuple[int, ...], entry: CacheEntry, match_len: int) -> bool:
        if not self.cache_enabled:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        if not getattr(self.backend, 'supports_external_kv', True):
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        total_tokens = len(tokens)
        suffix_len = max(total_tokens - match_len, 0)
        self.engine_metrics['reuse_attempts'] += 1
        if match_len < self.tuning.min_reuse_prefix_tokens:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        if total_tokens > 0 and match_len / total_tokens < self.tuning.min_prefix_coverage_ratio:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        estimated_saved = self._estimate_saved_ms(total_tokens, match_len, suffix_len)
        if estimated_saved < self.tuning.min_estimated_saved_ms:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        self.engine_metrics['estimated_tokens_saved_total'] = int(self.engine_metrics.get('estimated_tokens_saved_total', 0)) + int(match_len)
        self.engine_metrics['saved_latency_estimate_ms'] = float(self.engine_metrics.get('saved_latency_estimate_ms', 0.0)) + float(max(estimated_saved, 0.0))
        return True

    def _prefill_full(self, tokens: Tuple[int, ...]):
        out = self.backend.prefill(tokens)
        self._update_full_cost_stats(len(tokens), out.latency_ms)
        self.engine_metrics['full_prefill_latency_total_ms'] = float(self.engine_metrics.get('full_prefill_latency_total_ms', 0.0)) + float(out.latency_ms)
        return out

    def _empty_prefill_result(self, kv_cache):
        class _EmptyPrefillResult:
            def __init__(self, past_key_values):
                self.kv_cache = past_key_values
                self.latency_ms = 0.0
                self.memory_bytes = 0
                self.device = None
                self.gpu_utilization_pct = None

        return _EmptyPrefillResult(kv_cache)

    def _prefill_with_cache_fallback(self, tokens: Tuple[int, ...], entry: CacheEntry, match_len: int):
        suffix = tokens[match_len:]
        kv_cache = self.backend.prepare_past_key_values(entry.kv_cache)
        if len(suffix) == 0:
            self.engine_metrics['reuse_successes'] += 1
            self._update_reuse_cost_stats(0, 0.0, match_len)
            return self._empty_prefill_result(kv_cache), 0, True
        try:
            out = self.backend.prefill(suffix, past_key_values=kv_cache)
            self.engine_metrics['reuse_successes'] += 1
            self._update_reuse_cost_stats(len(suffix), out.latency_ms, match_len)
            return out, len(suffix), True
        except Exception:
            self.bank.remove(entry.prefix_tokens)
            self.engine_metrics['reuse_failures'] += 1
            self.bank.add_metric('reuse_failures', 1)
            out = self._prefill_full(tokens)
            return out, len(tokens), False


class NoCacheEngine(BaseEngine):
    def __init__(self, backend: Backend, max_memory_mb: int = 256):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb, name='no_cache')
        self.cache_enabled = False
        self.engine_metrics['cache_active_final'] = False

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...]) -> RequestResult:
        self.bank.observe_query(tokens)
        out = self._prefill_full(tokens)
        return self._record(
            RequestResult(
                request_id=request_id,
                latency_ms=out.latency_ms,
                matched_prefix_length=0,
                tokens_recomputed=len(tokens),
                was_cache_hit=False,
                was_speculative_hit=False,
                gpu_utilization_pct=out.gpu_utilization_pct,
            )
        )




class NativePrefixCachingEngine(BaseEngine):
    def __init__(self, backend: Backend, max_memory_mb: int = 256):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb, name='native_prefix_cache')
        self.engine_metrics['cache_active_final'] = True

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...]) -> RequestResult:
        self.bank.observe_query(tokens)
        out = self._prefill_full(tokens)
        return self._record(
            RequestResult(
                request_id=request_id,
                latency_ms=out.latency_ms,
                matched_prefix_length=0,
                tokens_recomputed=len(tokens),
                was_cache_hit=False,
                was_speculative_hit=False,
                gpu_utilization_pct=out.gpu_utilization_pct,
            )
        )

class ReactivePrefixCacheEngine(BaseEngine):
    def __init__(self, backend: Backend, max_memory_mb: int = 256):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb, name='reactive_prefix_cache')

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...]) -> RequestResult:
        self.bank.observe_query(tokens)
        match = self.bank.peek_match(tokens)

        if match is None:
            self.bank.record_miss()
            out = self._prefill_full(tokens)
            store_latency_ms = self._store_reactive_prefix(tokens, tier='cpu') if self.cache_enabled else 0.0
            return self._record(
                RequestResult(
                    request_id=request_id,
                    latency_ms=out.latency_ms + store_latency_ms,
                    matched_prefix_length=0,
                    tokens_recomputed=len(tokens),
                    was_cache_hit=False,
                    was_speculative_hit=False,
                    gpu_utilization_pct=out.gpu_utilization_pct,
                )
            )

        key, entry, match_len = match
        if not self._should_attempt_cache_use(tokens, entry, match_len):
            self.bank.record_miss()
            out = self._prefill_full(tokens)
            return self._record(
                RequestResult(
                    request_id=request_id,
                    latency_ms=out.latency_ms,
                    matched_prefix_length=0,
                    tokens_recomputed=len(tokens),
                    was_cache_hit=False,
                    was_speculative_hit=False,
                    gpu_utilization_pct=out.gpu_utilization_pct,
                )
            )

        admitted_entry = self.bank.admit_match(key)
        if admitted_entry is None:
            self.bank.record_miss()
            out = self._prefill_full(tokens)
            return self._record(
                RequestResult(
                    request_id=request_id,
                    latency_ms=out.latency_ms,
                    matched_prefix_length=0,
                    tokens_recomputed=len(tokens),
                    was_cache_hit=False,
                    was_speculative_hit=False,
                    gpu_utilization_pct=out.gpu_utilization_pct,
                )
            )

        out, recomputed_tokens, cache_hit = self._prefill_with_cache_fallback(tokens, admitted_entry, match_len)
        return self._record(
            RequestResult(
                request_id=request_id,
                latency_ms=out.latency_ms,
                matched_prefix_length=match_len if cache_hit else 0,
                tokens_recomputed=recomputed_tokens,
                was_cache_hit=cache_hit,
                was_speculative_hit=admitted_entry.was_speculative if cache_hit else False,
                cache_tier=admitted_entry.tier if cache_hit else None,
                gpu_utilization_pct=out.gpu_utilization_pct,
            )
        )



class StrictReactivePrefixCacheEngine(ReactivePrefixCacheEngine):
    def __init__(self, backend: Backend, max_memory_mb: int = 256):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb)
        self.name = 'strict_reactive_prefix_cache'
        self.tuning.min_reuse_prefix_tokens = max(self.tuning.min_reuse_prefix_tokens, 24)
        self.tuning.min_store_prefix_tokens = max(self.tuning.min_store_prefix_tokens, 20)
        self.tuning.min_prefix_coverage_ratio = max(self.tuning.min_prefix_coverage_ratio, 0.35)
        self.tuning.min_estimated_saved_ms = max(self.tuning.min_estimated_saved_ms, 14.0 if not self.backend.device.startswith('cuda') else 10.0)


class GreedyPrefixCacheEngine(ReactivePrefixCacheEngine):
    def __init__(self, backend: Backend, max_memory_mb: int = 256):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb)
        self.name = 'greedy_prefix_cache'
        self.tuning.min_store_prefix_tokens = self.bank.min_match_length
        self.tuning.min_reuse_prefix_tokens = self.bank.min_match_length
        self.tuning.min_estimated_saved_ms = 0.0
        self.tuning.min_prefix_coverage_ratio = 0.0
        self.tuning.max_cacheable_prefix_tokens = max(self.tuning.max_cacheable_prefix_tokens, 128)

    def _reactive_prefix_len(self, tokens: Tuple[int, ...]) -> int:
        return min(len(tokens), self.tuning.max_cacheable_prefix_tokens)

    def _should_store_reactive_prefix(self, tokens: Tuple[int, ...], prefix_len: int) -> bool:
        self.engine_metrics['store_attempts'] += 1
        if not getattr(self.backend, 'supports_external_kv', True):
            self.engine_metrics['store_skips'] += 1
            return False
        if len(tokens) < self.bank.min_match_length:
            self.engine_metrics['store_skips'] += 1
            return False
        return True

    def _should_attempt_cache_use(self, tokens: Tuple[int, ...], entry: CacheEntry, match_len: int) -> bool:
        if not self.cache_enabled:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        if not getattr(self.backend, 'supports_external_kv', True):
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        self.engine_metrics['reuse_attempts'] += 1
        if match_len < self.bank.min_match_length:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        total_tokens = len(tokens)
        suffix_len = max(total_tokens - match_len, 0)
        self.engine_metrics['estimated_tokens_saved_total'] = int(self.engine_metrics.get('estimated_tokens_saved_total', 0)) + int(match_len)
        self.engine_metrics['saved_latency_estimate_ms'] = float(self.engine_metrics.get('saved_latency_estimate_ms', 0.0)) + float(max(self._estimate_saved_ms(total_tokens, match_len, suffix_len), 0.0))
        return True


class FrequencySpeculativeEngine(ReactivePrefixCacheEngine):
    def __init__(self, backend: Backend, max_memory_mb: int = 256, speculative_k: int = 4, idle_threshold_ms: float = 60.0):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb)
        self.name = 'frequency_speculative'
        self.policy = FrequencyPolicy(min_frequency=0.22, min_prefix_len=max(self.tuning.min_store_prefix_tokens, 12), max_prefix_len=96, min_observations=3)
        self.speculative_k = speculative_k
        self.idle_threshold_ms = idle_threshold_ms
        self.last_request_time = time.time()
        self.serving_event = threading.Event()
        self.stop_event = threading.Event()
        self.speculative_log: List[Dict] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...]) -> RequestResult:
        self.serving_event.set()
        self.last_request_time = time.time()
        try:
            return super().serve_tokens(request_id, tokens)
        finally:
            self.serving_event.clear()

    def _speculation_allowed(self, prefix_tokens: Tuple[int, ...]) -> bool:
        if not getattr(self.backend, 'supports_external_kv', True):
            return False
        if len(prefix_tokens) < self.tuning.min_store_prefix_tokens:
            return False
        estimated_cost = self._estimate_full_cost_ms(len(prefix_tokens))
        return estimated_cost >= self.tuning.min_estimated_saved_ms

    def _loop(self) -> None:
        if not getattr(self.backend, 'supports_external_kv', True):
            return
        while not self.stop_event.is_set():
            time.sleep(0.005)
            idle_ms = (time.time() - self.last_request_time) * 1000.0
            if self.serving_event.is_set() or idle_ms < self.idle_threshold_ms:
                continue

            decisions = self.policy.rank(self.bank, budget_k=self.speculative_k, prefer_gpu=False)
            for decision in decisions:
                if self.serving_event.is_set() or self.stop_event.is_set():
                    break
                if self.bank.contains(decision.prefix_tokens) or not self._speculation_allowed(decision.prefix_tokens):
                    continue

                out = self.backend.prefill(decision.prefix_tokens)
                self._update_full_cost_stats(len(decision.prefix_tokens), out.latency_ms)
                stored = self.bank.store(
                    decision.prefix_tokens,
                    self.backend.move_kv_cache(out.kv_cache, 'cpu'),
                    out.latency_ms,
                    out.memory_bytes,
                    is_speculative=True,
                    tier='cpu',
                )
                if not stored:
                    continue
                self.speculative_log.append(
                    {
                        'policy': 'frequency',
                        'score': decision.score,
                        'latency_ms': out.latency_ms,
                        'target_tier': 'cpu',
                    }
                )

    def shutdown(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.finalize()


class ShadowKVEngine(ReactivePrefixCacheEngine):
    """Reactive prefix caching plus idle-time precompute."""

    def __init__(
        self,
        backend: Backend,
        max_memory_mb: int = 256,
        policy: SpeculationPolicy | None = None,
        speculative_k: int = 4,
        idle_threshold_ms: float = 60.0,
        enable_gpu_tier: bool = True,
    ):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb)
        self.name = 'shadow_kv'
        self.policy = policy or CostAwareSlackPolicy()
        self.speculative_k = speculative_k
        self.idle_threshold_ms = idle_threshold_ms
        self.enable_gpu_tier = enable_gpu_tier and backend.device.startswith('cuda')
        self._cpu_mode = not backend.device.startswith('cuda')
        self.last_request_time = time.time()
        self.serving_event = threading.Event()
        self.stop_event = threading.Event()
        self.speculative_log: List[Dict] = []
        if getattr(backend, 'backend_name', 'fake') == 'fake':
            self._min_requests_before_speculation = 6
        elif self._cpu_mode:
            self._min_requests_before_speculation = 8
        else:
            self._min_requests_before_speculation = 6
        self._speculation_cooldown_until = 0.0
        self._recent_speculative_net_values: deque[float] = deque(maxlen=6)
        self._recent_request_window: deque[Tuple[int, int, bool]] = deque(maxlen=16 if self._cpu_mode else 12)
        self._max_pending_speculative = 2 if self._cpu_mode else 1
        self._effective_speculative_k = speculative_k
        self._last_controller_metrics = {
            'speculative_hits': 0,
            'wasted_precomputes': 0,
            'wasted_compute_ms': 0.0,
            'useful_speculative_savings_ms': 0.0,
        }
        self.engine_metrics.update(
            {
                'speculation_cooldown_events': 0,
                'speculation_paused_ticks': 0,
                'speculation_enabled_final': True,
                'effective_speculative_k_final': speculative_k,
                'recent_reuse_density_final': 0.0,
                'recent_hit_rate_final': 0.0,
            }
        )
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _after_record(self, result: RequestResult) -> None:
        total_tokens = max(result.tokens_recomputed + result.matched_prefix_length, 1)
        self._recent_request_window.append((total_tokens, result.matched_prefix_length, result.was_cache_hit))

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...]) -> RequestResult:
        self.serving_event.set()
        self.last_request_time = time.time()
        try:
            self.bank.observe_query(tokens)
            match = self.bank.peek_match(tokens)
            if match is None:
                self.bank.record_miss()
                out = self._prefill_full(tokens)
                store_latency_ms = self._store_reactive_prefix(tokens, tier='gpu' if self.enable_gpu_tier else 'cpu')
                result = RequestResult(
                    request_id=request_id,
                    latency_ms=out.latency_ms + store_latency_ms,
                    matched_prefix_length=0,
                    tokens_recomputed=len(tokens),
                    was_cache_hit=False,
                    was_speculative_hit=False,
                    gpu_utilization_pct=out.gpu_utilization_pct,
                )
            else:
                key, entry, match_len = match
                if not self._should_attempt_cache_use(tokens, entry, match_len):
                    self.bank.record_miss()
                    out = self._prefill_full(tokens)
                    result = RequestResult(
                        request_id=request_id,
                        latency_ms=out.latency_ms,
                        matched_prefix_length=0,
                        tokens_recomputed=len(tokens),
                        was_cache_hit=False,
                        was_speculative_hit=False,
                        gpu_utilization_pct=out.gpu_utilization_pct,
                    )
                else:
                    admitted_entry = self.bank.admit_match(key)
                    if admitted_entry is None:
                        self.bank.record_miss()
                        out = self._prefill_full(tokens)
                        result = RequestResult(
                            request_id=request_id,
                            latency_ms=out.latency_ms,
                            matched_prefix_length=0,
                            tokens_recomputed=len(tokens),
                            was_cache_hit=False,
                            was_speculative_hit=False,
                            gpu_utilization_pct=out.gpu_utilization_pct,
                        )
                    else:
                        out, recomputed_tokens, cache_hit = self._prefill_with_cache_fallback(tokens, admitted_entry, match_len)
                        if (
                            cache_hit
                            and self.enable_gpu_tier
                            and admitted_entry.tier == 'cpu'
                            and admitted_entry.hit_count >= self.tuning.promote_after_hits
                            and len(admitted_entry.prefix_tokens) <= self.tuning.gpu_promotion_max_prefix_tokens
                        ):
                            promoted = self.backend.move_kv_cache(admitted_entry.kv_cache, self.backend.device)
                            if self.bank.promote(admitted_entry.prefix_tokens, promoted, admitted_entry.memory_bytes, new_tier='gpu'):
                                admitted_entry.tier = 'gpu'

                        result = RequestResult(
                            request_id=request_id,
                            latency_ms=out.latency_ms,
                            matched_prefix_length=match_len if cache_hit else 0,
                            tokens_recomputed=recomputed_tokens,
                            was_cache_hit=cache_hit,
                            was_speculative_hit=admitted_entry.was_speculative if cache_hit else False,
                            cache_tier=admitted_entry.tier if cache_hit else None,
                            gpu_utilization_pct=out.gpu_utilization_pct,
                        )

            return self._record(result)
        finally:
            self.serving_event.clear()

    def _speculation_allowed(self, decision_prefix: Tuple[int, ...], decision_benefit_ms: float, decision_cost_ms: float) -> bool:
        if len(decision_prefix) < self.tuning.min_store_prefix_tokens:
            return False
        estimated_cost = self._estimate_full_cost_ms(len(decision_prefix))
        recent_support = self.bank.recent_prefix_support(decision_prefix)
        recent_streak = self.bank.recent_prefix_streak(decision_prefix)
        if recent_support <= 0.0 and recent_streak == 0:
            return False
        if self._cpu_mode:
            support_gate = recent_support >= 0.08 or recent_streak >= 2
        else:
            support_gate = recent_support >= 0.10 or recent_streak >= 2
        return (
            decision_benefit_ms >= max(decision_cost_ms, 0.0)
            and estimated_cost >= self.tuning.min_estimated_saved_ms
            and support_gate
        )

    def _refresh_speculation_controller(self) -> bool:
        now = time.time()
        snapshot = self.bank.snapshot_metrics()
        delta_useful = float(snapshot.get('useful_speculative_savings_ms', 0.0)) - float(self._last_controller_metrics['useful_speculative_savings_ms'])
        delta_waste = float(snapshot.get('wasted_compute_ms', 0.0)) - float(self._last_controller_metrics['wasted_compute_ms'])
        delta_hits = int(snapshot.get('speculative_hits', 0)) - int(self._last_controller_metrics['speculative_hits'])
        delta_wasted = int(snapshot.get('wasted_precomputes', 0)) - int(self._last_controller_metrics['wasted_precomputes'])

        if delta_hits > 0 or delta_wasted > 0:
            self._recent_speculative_net_values.append(delta_useful - delta_waste)
        if delta_wasted > 0 and delta_useful <= delta_waste:
            self._speculation_cooldown_until = max(self._speculation_cooldown_until, now + max(self.idle_threshold_ms / 1000.0, 0.25))
            self.engine_metrics['speculation_cooldown_events'] = int(self.engine_metrics.get('speculation_cooldown_events', 0)) + 1

        self._last_controller_metrics = {
            'speculative_hits': int(snapshot.get('speculative_hits', 0)),
            'wasted_precomputes': int(snapshot.get('wasted_precomputes', 0)),
            'wasted_compute_ms': float(snapshot.get('wasted_compute_ms', 0.0)),
            'useful_speculative_savings_ms': float(snapshot.get('useful_speculative_savings_ms', 0.0)),
        }

        requests_seen = int(self.engine_metrics.get('requests_seen', 0))
        pending_speculative = int(snapshot.get('pending_speculative_entries', 0))
        recent_net = sum(self._recent_speculative_net_values)
        recent_count = len(self._recent_speculative_net_values)
        recent_total_tokens = sum(total for total, _, _ in self._recent_request_window)
        recent_reused_tokens = sum(reused for _, reused, _ in self._recent_request_window)
        recent_hits = sum(1 for _, _, was_hit in self._recent_request_window if was_hit)
        recent_request_count = len(self._recent_request_window)
        recent_reuse_density = recent_reused_tokens / max(recent_total_tokens, 1)
        recent_hit_rate = recent_hits / max(recent_request_count, 1)
        self.engine_metrics['recent_reuse_density_final'] = float(recent_reuse_density)
        self.engine_metrics['recent_hit_rate_final'] = float(recent_hit_rate)

        if requests_seen < self._min_requests_before_speculation:
            self._effective_speculative_k = 0
        elif now < self._speculation_cooldown_until:
            self._effective_speculative_k = 0
        elif pending_speculative >= self._max_pending_speculative:
            self._effective_speculative_k = 0
        elif recent_request_count >= 6 and recent_reuse_density < 0.04 and recent_hit_rate < 0.10:
            self._effective_speculative_k = 0
        elif recent_count >= 2 and recent_net <= 0.0:
            self._effective_speculative_k = 0
        elif recent_count >= 1 and recent_net < 12.0:
            self._effective_speculative_k = 1
        elif self._cpu_mode and (recent_hit_rate >= 0.30 or recent_reuse_density >= 0.10):
            self._effective_speculative_k = min(max(self.speculative_k, 1), 2)
        elif (not self._cpu_mode) and (recent_hit_rate >= 0.10 or recent_reuse_density >= 0.04):
            self._effective_speculative_k = min(max(self.speculative_k, 1), 2)
        elif recent_hit_rate < 0.20 and recent_reuse_density < 0.08:
            self._effective_speculative_k = 1
        else:
            self._effective_speculative_k = self.speculative_k

        speculation_enabled = self._effective_speculative_k > 0
        if not speculation_enabled:
            self.engine_metrics['speculation_paused_ticks'] = int(self.engine_metrics.get('speculation_paused_ticks', 0)) + 1
        self.engine_metrics['speculation_enabled_final'] = speculation_enabled
        self.engine_metrics['effective_speculative_k_final'] = self._effective_speculative_k
        return speculation_enabled

    def _loop(self) -> None:
        if not getattr(self.backend, 'supports_external_kv', True):
            return
        while not self.stop_event.is_set():
            time.sleep(0.005)
            idle_ms = (time.time() - self.last_request_time) * 1000.0
            if self.serving_event.is_set() or idle_ms < self.idle_threshold_ms:
                continue
            if not self._refresh_speculation_controller():
                continue

            prefer_gpu = self.enable_gpu_tier and idle_ms >= self.idle_threshold_ms
            decisions = self.policy.rank(self.bank, budget_k=self._effective_speculative_k, prefer_gpu=prefer_gpu)

            for decision in decisions:
                if self.serving_event.is_set() or self.stop_event.is_set():
                    break
                if self.bank.contains(decision.prefix_tokens):
                    continue
                if not self._speculation_allowed(decision.prefix_tokens, decision.expected_benefit_ms, decision.expected_cost_ms):
                    continue

                out = self.backend.prefill(decision.prefix_tokens)
                self._update_full_cost_stats(len(decision.prefix_tokens), out.latency_ms)
                recent_streak = self.bank.recent_prefix_streak(decision.prefix_tokens)
                target_tier = decision.target_tier if prefer_gpu else 'cpu'
                if self.enable_gpu_tier and recent_streak >= 2:
                    target_tier = 'gpu'
                device_target = self._device_target_for_tier(target_tier)
                cache_obj = out.kv_cache if device_target == self.backend.device else self.backend.move_kv_cache(out.kv_cache, device_target)

                stored = self.bank.store(
                    decision.prefix_tokens,
                    cache_obj,
                    out.latency_ms,
                    out.memory_bytes,
                    is_speculative=True,
                    tier=target_tier,
                )
                if not stored:
                    continue
                self.speculative_log.append(
                    {
                        'policy': 'template_aware_cost_aware_slack',
                        'score': decision.score,
                        'latency_ms': out.latency_ms,
                        'target_tier': target_tier,
                        'expected_benefit_ms': decision.expected_benefit_ms,
                        'expected_cost_ms': decision.expected_cost_ms,
                    }
                )

    def shutdown(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.finalize()


def maybe_shutdown(engine: BaseEngine) -> None:
    if hasattr(engine, 'shutdown'):
        engine.shutdown()  # type: ignore[attr-defined]
    else:
        engine.finalize()

    if hasattr(engine, 'bank') and hasattr(engine.bank, 'finalize_run'):
        engine.bank.finalize_run()


def summarize_engine(engine: BaseEngine) -> Dict:
    speculative_log = getattr(engine, 'speculative_log', [])
    return summarize_run(
        latencies=engine.latencies,
        gpu_utils=engine.gpu_utils,
        bank_metrics=engine.bank.snapshot_metrics(),
        speculative_precomputes=len(speculative_log),
        speculative_cost_ms=sum(x['latency_ms'] for x in speculative_log),
        total_wall_time_s=max(engine.end_time - engine.start_time, 1e-9),
        extra_metrics=engine.engine_metrics,
    ).to_dict()


def compare_named_runs(engines: List[BaseEngine]) -> Dict[str, Dict]:
    out = {}
    for engine in engines:
        out[engine.name] = summarize_engine(engine)
    if 'no_cache' in out:
        baseline = out['no_cache']
        for _, summary in out.items():
            summary['speedup_vs_no_cache_mean'] = baseline['mean_latency_ms'] / max(summary['mean_latency_ms'], 1e-9)
            summary['speedup_vs_no_cache_p95'] = baseline['p95_latency_ms'] / max(summary['p95_latency_ms'], 1e-9)
    return out
