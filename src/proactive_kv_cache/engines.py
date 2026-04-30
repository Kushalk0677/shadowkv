from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .cache import CacheEntry, TieredStateBank
from .metrics import summarize_run
from .models import Backend, estimate_past_key_values_bytes
from .policy import CostAwareSlackPolicy, FrequencyPolicy, SpeculationPolicy
from .semantic import SemanticKVIndex, longest_common_prefix_len, token_entropy
from .controller import AdaptiveReuseController, ReusePlan


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
            'speculative_overlap_ms': 0.0,
            'speculative_overlap_events': 0,
            'bootstrap_store_deferrals': 0,
            'speculative_useful_savings_ms': 0.0,
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
        self.policy_trace_rows: List[Dict] = []
        self.trace_enabled = False
        self._current_trace_context: Dict | None = None
        self._current_trace_plan: Dict | None = None
        self._current_semantic_match: Dict | None = None

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
                gpu_promotion_max_prefix_tokens=128,
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
        if self.name in ('no_cache', 'native_prefix_cache') or bool(self.engine_metrics.get('native_runtime_cache_baseline', False)):
            return
        requests_seen = int(self.engine_metrics.get('requests_seen', 0))
        attempts = int(self.engine_metrics.get('reuse_attempts', 0))
        successes = int(self.engine_metrics.get('reuse_successes', 0))
        bypassed = int(self.engine_metrics.get('bypassed_matches', 0))
        store_successes = int(self.engine_metrics.get('store_successes', 0))
        min_requests = int(self.engine_metrics.get('cache_disable_min_requests', 10))
        min_attempts = int(self.engine_metrics.get('cache_disable_min_attempts', 8))
        check_every = max(int(self.engine_metrics.get('cache_disable_check_every', 8)), 1)
        bank_snapshot = self.bank.snapshot_metrics()
        if requests_seen >= 4:
            all_misses = int(bank_snapshot.get('hits', 0)) == 0 and int(bank_snapshot.get('misses', 0)) >= requests_seen
            if all_misses and attempts == 0 and store_successes >= 3:
                self._set_cache_disabled('no_prefix_reuse_early')
                return
        if requests_seen < min_requests or attempts < min_attempts:
            return
        if requests_seen % check_every != 0 and not (successes == 0 and bypassed >= min_attempts):
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
        if recompute > 0 and reused / max(recompute + reused, 1) < 0.03 and bypass_rate >= 0.75 and not self._has_meaningful_absolute_reuse(reused, successes):
            self._set_cache_disabled('low_reuse_density')
            return

    def _has_meaningful_absolute_reuse(self, reused_tokens: int, hit_count: int = 0) -> bool:
        token_gate = max(self.tuning.min_reuse_prefix_tokens * 4, 64)
        return reused_tokens >= token_gate or hit_count >= 2

    def _should_defer_reactive_store(self, tokens: Tuple[int, ...], prefix_len: int, metadata: Dict | None = None) -> bool:
        return False

    def finalize(self) -> None:
        self._mark_cache_active()
        self.end_time = time.perf_counter()

    def _append_policy_trace(self, result: RequestResult) -> None:
        if not getattr(self, 'trace_enabled', False):
            return
        context = dict(self._current_trace_context or {})
        metadata = context.pop('metadata', {})
        snapshot = self.bank.snapshot_metrics()
        token_count = int(context.get('token_count', result.tokens_recomputed + result.matched_prefix_length))
        row = {
            'engine': self.name,
            'request_id': result.request_id,
            'latency_ms': float(result.latency_ms),
            'token_count': token_count,
            'matched_prefix_length': int(result.matched_prefix_length),
            'tokens_recomputed': int(result.tokens_recomputed),
            'was_cache_hit': bool(result.was_cache_hit),
            'was_speculative_hit': bool(result.was_speculative_hit),
            'cache_tier': result.cache_tier,
            'gpu_utilization_pct': result.gpu_utilization_pct,
            'cache_enabled_before': context.get('cache_enabled_before'),
            'cache_enabled_after': bool(self.cache_enabled),
            'auto_disabled_reason': self.engine_metrics.get('auto_disabled_reason'),
            'shared_prefix_hint_tokens': context.get('shared_prefix_hint_tokens'),
            'reuse_attempts_cumulative': int(self.engine_metrics.get('reuse_attempts', 0)),
            'reuse_successes_cumulative': int(self.engine_metrics.get('reuse_successes', 0)),
            'reuse_failures_cumulative': int(self.engine_metrics.get('reuse_failures', 0)),
            'store_attempts_cumulative': int(self.engine_metrics.get('store_attempts', 0)),
            'store_successes_cumulative': int(self.engine_metrics.get('store_successes', 0)),
            'store_skips_cumulative': int(self.engine_metrics.get('store_skips', 0)),
            'bypassed_matches_cumulative': int(self.engine_metrics.get('bypassed_matches', 0)),
            'bank_hits_cumulative': int(snapshot.get('hits', 0)),
            'bank_misses_cumulative': int(snapshot.get('misses', 0)),
            'bank_hit_rate_cumulative': float(snapshot.get('hit_rate', 0.0)),
            'wasted_precomputes_cumulative': int(snapshot.get('wasted_precomputes', 0)),
            'wasted_compute_ms_cumulative': float(snapshot.get('wasted_compute_ms', 0.0)),
            'ewma_full_ms_per_token_before': context.get('ewma_full_ms_per_token_before'),
            'ewma_reuse_ms_per_token_before': context.get('ewma_reuse_ms_per_token_before'),
            'ewma_cache_reuse_overhead_ms_before': context.get('ewma_cache_reuse_overhead_ms_before'),
            'metadata': metadata,
            'label_reused': 1 if result.was_cache_hit else 0,
            'label_bypassed_or_miss': 1 if not result.was_cache_hit else 0,
        }
        if self._current_trace_plan:
            row.update(self._current_trace_plan)
        if self._current_semantic_match:
            row.update(self._current_semantic_match)
        if token_count > 0:
            row['actual_reuse_fraction'] = float(result.matched_prefix_length / max(token_count, 1))
        self.policy_trace_rows.append(row)

    def _record(self, result: RequestResult) -> RequestResult:
        self.latencies.append(result.latency_ms)
        self.results.append(result)
        self.gpu_utils.append(result.gpu_utilization_pct)
        self.engine_metrics['recompute_tokens_total'] = int(self.engine_metrics.get('recompute_tokens_total', 0)) + int(result.tokens_recomputed)
        if result.was_cache_hit and result.matched_prefix_length > 0:
            self.engine_metrics['reused_prefix_tokens_total'] = int(self.engine_metrics.get('reused_prefix_tokens_total', 0)) + int(result.matched_prefix_length)
        self._after_record(result)
        self._on_request_finish()
        snapshot = self.bank.snapshot_metrics()
        self.engine_metrics['speculative_useful_savings_ms'] = float(snapshot.get('useful_speculative_savings_ms', 0.0))
        self._append_policy_trace(result)
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

    def _shared_prefix_hint(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> int | None:
        if not metadata:
            return None
        hint = metadata.get('shared_prefix_hint_tokens')
        if hint is None:
            return None
        try:
            hint_len = int(hint)
        except (TypeError, ValueError):
            return None
        if hint_len < self.bank.min_match_length:
            return None
        return min(len(tokens), hint_len)

    def _has_scaffold_hint(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> bool:
        hint_len = self._shared_prefix_hint(tokens, metadata)
        if hint_len is None:
            return False
        prompt_mode = str((metadata or {}).get('prompt_mode', '')).strip().lower()
        if prompt_mode in {'templated', 'rag'}:
            return True
        return bool((metadata or {}).get('shared_prefix_text'))

    def _tracked_prefix_lengths(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> List[int]:
        hint_len = self._shared_prefix_hint(tokens, metadata)
        if hint_len is None:
            return self.bank.default_prefix_lengths(tokens)
        if self._has_scaffold_hint(tokens, metadata):
            return [hint_len]
        hint_tokens = tokens[:hint_len]
        lengths = set(self.bank.default_prefix_lengths(hint_tokens))
        lengths.add(hint_len)
        return sorted(length for length in lengths if self.bank.min_match_length <= length <= hint_len)

    def _safe_trace_metadata(self, metadata: Dict | None) -> Dict:
        if not metadata:
            return {}
        allowed = {
            'arrival_time', 'dataset', 'prompt_mode', 'shared_prefix_hint_tokens', 'shared_prefix_text',
            'semantic_equivalence_key', 'semantic_family', 'paraphrase_variant', 'semantic_variant_count',
            'semantic_anchor_text', 'label', 'category', 'source', 'split', 'template_id', 'request_index',
        }
        out = {}
        for key, value in metadata.items():
            if key not in allowed:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
            else:
                out[key] = str(value)
        for key in ('shared_prefix_text', 'semantic_anchor_text'):
            if isinstance(out.get(key), str) and len(out[key]) > 240:
                out[key] = out[key][:240] + '...'
        return out

    def _observe_request(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> None:
        self._current_trace_context = {
            'token_count': len(tokens),
            'shared_prefix_hint_tokens': self._shared_prefix_hint(tokens, metadata),
            'cache_enabled_before': bool(self.cache_enabled),
            'ewma_full_ms_per_token_before': float(self.ewma_full_ms_per_token),
            'ewma_reuse_ms_per_token_before': float(self.ewma_reuse_ms_per_token),
            'ewma_cache_reuse_overhead_ms_before': float(self.ewma_cache_reuse_overhead_ms),
            'metadata': self._safe_trace_metadata(metadata),
        }
        self._current_trace_plan = None
        self._current_semantic_match = None
        tracked_prefix_lengths = self._tracked_prefix_lengths(tokens, metadata)
        self.bank.observe_query(
            tokens,
            tracked_prefix_lengths=tracked_prefix_lengths,
            reusable_prefix_limit=self._shared_prefix_hint(tokens, metadata),
            observed_at=float(metadata['arrival_time']) if metadata and metadata.get('arrival_time') is not None else None,
        )

    def _request_cacheable_prefix_limit(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> int:
        hint_len = self._shared_prefix_hint(tokens, metadata)
        if hint_len is None:
            return min(len(tokens), self.tuning.max_cacheable_prefix_tokens)
        return min(hint_len, self.tuning.max_cacheable_prefix_tokens)

    def _estimate_full_cost_ms(self, token_count: int) -> float:
        if token_count <= 0:
            return 0.0
        if self.ewma_full_ms_per_token > 0.0:
            return self.ewma_full_ms_per_token * token_count
        return self.backend.estimate_prefill_cost_ms(token_count)

    def _materialize_cache_for_tier(self, kv_cache, prefix_len: int, tier: str) -> tuple[object, float, int]:
        device_target = self._device_target_for_tier(tier)
        cache_obj = kv_cache
        move_latency_ms = 0.0
        if device_target != self.backend.device:
            move_start = time.perf_counter()
            cache_obj = self.backend.move_kv_cache(kv_cache, device_target)
            move_latency_ms = (time.perf_counter() - move_start) * 1000.0
        memory_bytes = estimate_past_key_values_bytes(cache_obj)
        if memory_bytes <= 0:
            memory_bytes = self.backend.estimate_kv_cache_bytes(prefix_len)
        return cache_obj, move_latency_ms, memory_bytes

    def _estimate_saved_ms(self, total_tokens: int, match_len: int, suffix_len: int) -> float:
        matched_cost = self._estimate_full_cost_ms(match_len)
        suffix_cost = self._estimate_full_cost_ms(suffix_len)
        gross_saved = max(matched_cost, 0.0)
        penalty = self.ewma_cache_reuse_overhead_ms + 0.03 * max(suffix_len, 0)
        if total_tokens > 0 and match_len / total_tokens < self.tuning.min_prefix_coverage_ratio:
            penalty += 4.0
        return gross_saved - penalty - 0.10 * suffix_cost

    def _estimate_scaffold_saved_ms(self, total_tokens: int, match_len: int, suffix_len: int) -> float:
        matched_cost = self._estimate_full_cost_ms(match_len)
        suffix_cost = self._estimate_full_cost_ms(suffix_len)
        gross_saved = max(matched_cost, 0.0)
        # Shared prompt scaffolds are deliberately stable across requests, so
        # use a lighter penalty than the generic reactive path.
        penalty = (self.ewma_cache_reuse_overhead_ms * 0.35) + 0.01 * max(suffix_len, 0)
        return gross_saved - penalty - 0.03 * suffix_cost

    def _reactive_prefix_len(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> int:
        if self._has_scaffold_hint(tokens, metadata):
            return self._request_cacheable_prefix_limit(tokens, metadata)
        candidate_lengths = self._tracked_prefix_lengths(tokens, metadata)
        max_cacheable_prefix = self._request_cacheable_prefix_limit(tokens, metadata)
        best_len = min(len(tokens), max(self.bank.min_match_length, self.tuning.min_store_prefix_tokens))
        best_score = float('-inf')

        for length in candidate_lengths:
            if (
                length < self.bank.min_match_length
                or length > len(tokens)
                or length < self.tuning.min_store_prefix_tokens
                or length > max_cacheable_prefix
            ):
                continue
            prefix = tokens[:length]
            freq = self.bank.get_frequency(prefix)
            branching = self.bank.branching_factor(prefix)
            coverage = length / max(len(tokens), 1)
            score = (
                (freq * (1.0 + 0.05 * min(length, 24)))
                + (0.5 * coverage)
                + (0.04 * min(max(branching - 1, 0), 4))
                - (0.015 * max(length - 32, 0))
            )
            if score > best_score:
                best_score = score
                best_len = length

        return best_len

    def _should_store_reactive_prefix(self, tokens: Tuple[int, ...], prefix_len: int, metadata: Dict | None = None) -> bool:
        self.engine_metrics['store_attempts'] += 1
        if not getattr(self.backend, 'supports_external_kv', True):
            self.engine_metrics['store_skips'] += 1
            return False
        if len(tokens) < self.tuning.min_store_prefix_tokens:
            self.engine_metrics['store_skips'] += 1
            return False
        if prefix_len < self.tuning.min_store_prefix_tokens or prefix_len > self._request_cacheable_prefix_limit(tokens, metadata):
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

    def _should_defer_reactive_store(self, tokens: Tuple[int, ...], prefix_len: int, metadata: Dict | None = None) -> bool:
        return False

    def _store_reactive_prefix(self, tokens: Tuple[int, ...], tier: str = 'cpu', metadata: Dict | None = None) -> float:
        prefix_len = self._reactive_prefix_len(tokens, metadata)
        if not self._should_store_reactive_prefix(tokens, prefix_len, metadata):
            return 0.0
        if self._should_defer_reactive_store(tokens, prefix_len, metadata):
            self.engine_metrics['bootstrap_store_deferrals'] = int(self.engine_metrics.get('bootstrap_store_deferrals', 0)) + 1
            self.engine_metrics['store_skips'] += 1
            return 0.0
        prefix = tokens[:prefix_len]
        prefill = self.backend.prefill(prefix)
        self._update_full_cost_stats(prefix_len, prefill.latency_ms)
        cache_obj, move_latency_ms, memory_bytes = self._materialize_cache_for_tier(prefill.kv_cache, prefix_len, tier)
        total_latency_ms = float(prefill.latency_ms + move_latency_ms)
        stored = self.bank.store(prefix, cache_obj, total_latency_ms, memory_bytes, is_speculative=False, tier=tier)
        if stored:
            self.engine_metrics['store_successes'] += 1
            self.engine_metrics['store_latency_total_ms'] = float(self.engine_metrics.get('store_latency_total_ms', 0.0)) + total_latency_ms
            return total_latency_ms
        self.engine_metrics['store_skips'] += 1
        return 0.0

    def _should_attempt_cache_use(self, tokens: Tuple[int, ...], entry: CacheEntry, match_len: int, metadata: Dict | None = None) -> bool:
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
        hint_len = self._shared_prefix_hint(tokens, metadata)
        scaffold_match = hint_len is not None and match_len >= hint_len and self._has_scaffold_hint(tokens, metadata)
        self.engine_metrics['reuse_attempts'] += 1
        if match_len < self.tuning.min_reuse_prefix_tokens:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        if not scaffold_match and total_tokens > 0 and match_len / total_tokens < self.tuning.min_prefix_coverage_ratio:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        estimated_saved = (
            self._estimate_scaffold_saved_ms(total_tokens, match_len, suffix_len)
            if scaffold_match
            else self._estimate_saved_ms(total_tokens, match_len, suffix_len)
        )
        min_saved_gate = 0.0 if scaffold_match else self.tuning.min_estimated_saved_ms
        if estimated_saved < min_saved_gate:
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

    def _prefill_with_cache_fallback(self, tokens: Tuple[int, ...], entry: CacheEntry, match_len: int):
        suffix = tokens[match_len:]
        try:
            out = self.backend.prefill(suffix, past_key_values=entry.kv_cache)
        except (RuntimeError, ValueError, IndexError):
            self.bank.remove(entry.prefix_tokens)
            self.engine_metrics['reuse_failures'] += 1
            self.bank.add_metric('reuse_failures', 1)
            out = self._prefill_full(tokens)
            return out, len(tokens), False, 0
        if entry.kv_cache is not None and not out.used_past_key_values:
            self.bank.remove(entry.prefix_tokens)
            self.engine_metrics['reuse_failures'] += 1
            self.bank.add_metric('reuse_failures', 1)
            self.bank.add_metric('reuse_backend_fallbacks', 1)
            out = self._prefill_full(tokens)
            return out, len(tokens), False, 0
        actual_match_len = min(match_len, max(int(out.prepared_past_length), 0))
        self.engine_metrics['reuse_successes'] += 1
        recomputed_tokens = max(len(tokens) - actual_match_len, 0)
        self._update_reuse_cost_stats(recomputed_tokens, out.latency_ms, actual_match_len)
        return out, recomputed_tokens, True, actual_match_len


class NoCacheEngine(BaseEngine):
    def __init__(self, backend: Backend, max_memory_mb: int = 256):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb, name='no_cache')
        self.cache_enabled = False
        self.engine_metrics['cache_active_final'] = False

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...], metadata: Dict | None = None) -> RequestResult:
        self._observe_request(tokens, metadata)
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
        self._native_placeholder_bytes = 1

    def _store_native_placeholder(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> None:
        prefix_len = self._reactive_prefix_len(tokens, metadata)
        if prefix_len < self.bank.min_match_length:
            return
        prefix = tokens[:prefix_len]
        if self.bank.contains(prefix):
            return
        self.bank.store(prefix, None, generation_cost_ms=0.0, memory_bytes=self._native_placeholder_bytes, is_speculative=False, tier='cpu')

    def _should_attempt_cache_use(self, tokens: Tuple[int, ...], entry: CacheEntry, match_len: int, metadata: Dict | None = None) -> bool:
        if not self.cache_enabled:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        total_tokens = len(tokens)
        suffix_len = max(total_tokens - match_len, 0)
        hint_len = self._shared_prefix_hint(tokens, metadata)
        scaffold_match = hint_len is not None and match_len >= hint_len and self._has_scaffold_hint(tokens, metadata)
        self.engine_metrics['reuse_attempts'] += 1
        if match_len < self.tuning.min_reuse_prefix_tokens:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        if not scaffold_match and total_tokens > 0 and match_len / total_tokens < self.tuning.min_prefix_coverage_ratio:
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        estimated_saved = self._estimate_saved_ms(total_tokens, match_len, suffix_len)
        if estimated_saved < self.tuning.min_estimated_saved_ms * (0.5 if scaffold_match else 1.0):
            self.engine_metrics['bypassed_matches'] += 1
            self.bank.add_metric('bypassed_matches', 1)
            return False
        self.engine_metrics['estimated_tokens_saved_total'] = int(self.engine_metrics.get('estimated_tokens_saved_total', 0)) + int(match_len)
        self.engine_metrics['saved_latency_estimate_ms'] = float(self.engine_metrics.get('saved_latency_estimate_ms', 0.0)) + float(max(estimated_saved, 0.0))
        return True

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...], metadata: Dict | None = None) -> RequestResult:
        self._observe_request(tokens, metadata)
        match = self.bank.peek_match(tokens)
        matched_prefix_length = 0
        was_cache_hit = False
        if match is not None:
            _, entry, match_len = match
            if self._should_attempt_cache_use(tokens, entry, match_len, metadata=metadata):
                admitted_entry = self.bank.admit_match(match[0])
                if admitted_entry is not None:
                    matched_prefix_length = match_len
                    was_cache_hit = True
        out = self._prefill_full(tokens)
        self._store_native_placeholder(tokens, metadata)
        return self._record(
            RequestResult(
                request_id=request_id,
                latency_ms=out.latency_ms,
                matched_prefix_length=matched_prefix_length,
                tokens_recomputed=max(len(tokens) - matched_prefix_length, 0) if was_cache_hit else len(tokens),
                was_cache_hit=was_cache_hit,
                was_speculative_hit=False,
                gpu_utilization_pct=out.gpu_utilization_pct,
            )
        )


class RuntimeNativeCacheEngine(NativePrefixCachingEngine):
    """Named native-runtime cache baseline.

    These baselines represent runtime-managed prefix/KV caches such as vLLM APC,
    SGLang RadixAttention, and LMCache. The repository cannot force every
    external runtime through the in-process `Backend` interface, so this engine
    uses the same placeholder accounting as `NativePrefixCachingEngine` while
    preserving the runtime name in result JSON. When the backend itself exposes
    native caching, for example vLLM with prefix caching enabled, the measured
    latency comes from that backend path.
    """

    def __init__(
        self,
        backend: Backend,
        max_memory_mb: int = 256,
        name: str = 'runtime_native_cache',
        runtime_family: str = 'native_runtime',
    ):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb)
        self.name = name
        self.runtime_family = runtime_family
        self.engine_metrics.update(
            {
                'native_runtime_cache_baseline': True,
                'native_runtime_family': runtime_family,
                'admission_controller_enabled': False,
            }
        )


class AdmissionControlledRuntimeCacheEngine(RuntimeNativeCacheEngine):
    """Native runtime cache baseline gated by the ShadowKV++ admission policy."""

    def __init__(
        self,
        backend: Backend,
        max_memory_mb: int = 256,
        name: str = 'runtime_native_cache_shadowkv_plus',
        runtime_family: str = 'native_runtime',
        semantic_similarity_threshold: float = 0.58,
    ):
        super().__init__(
            backend=backend,
            max_memory_mb=max_memory_mb,
            name=name,
            runtime_family=runtime_family,
        )
        self.controller = AdaptiveReuseController(
            min_utility_ms=0.0,
            semantic_threshold=semantic_similarity_threshold,
            max_layer_reuse_ratio=0.55,
        )
        self.engine_metrics.update(
            {
                'admission_controller_enabled': True,
                'admission_plans_total': 0,
                'admission_allow_total': 0,
                'admission_bypass_total': 0,
                'admission_store_total': 0,
                'admission_store_bypass_total': 0,
                'admission_policy_net_utility_ms': 0.0,
                'admission_policy_expected_benefit_ms': 0.0,
                'admission_policy_expected_cost_ms': 0.0,
                'admission_policy_expected_waste_ms': 0.0,
            }
        )

    def _record_admission_plan(self, plan: ReusePlan) -> None:
        self.engine_metrics['admission_plans_total'] = int(self.engine_metrics.get('admission_plans_total', 0)) + 1
        allowed = plan.strategy == 'exact' and plan.score >= 0.0
        self.engine_metrics['admission_allow_total' if allowed else 'admission_bypass_total'] = int(
            self.engine_metrics.get('admission_allow_total' if allowed else 'admission_bypass_total', 0)
        ) + 1
        self.engine_metrics['admission_policy_net_utility_ms'] = float(self.engine_metrics.get('admission_policy_net_utility_ms', 0.0)) + float(plan.score)
        self.engine_metrics['admission_policy_expected_benefit_ms'] = float(self.engine_metrics.get('admission_policy_expected_benefit_ms', 0.0)) + float(plan.expected_benefit_ms)
        self.engine_metrics['admission_policy_expected_cost_ms'] = float(self.engine_metrics.get('admission_policy_expected_cost_ms', 0.0)) + float(plan.expected_cost_ms)
        self.engine_metrics['admission_policy_expected_waste_ms'] = float(self.engine_metrics.get('admission_policy_expected_waste_ms', 0.0)) + float(plan.expected_waste_ms)
        self._current_trace_plan = {
            'policy_strategy': plan.strategy,
            'policy_speculate_depth_tokens': int(plan.speculate_depth_tokens),
            'policy_reusable_prefix_tokens': int(plan.reusable_prefix_tokens),
            'policy_layer_reuse_ratio': float(plan.layer_reuse_ratio),
            'policy_expected_benefit_ms': float(plan.expected_benefit_ms),
            'policy_expected_cost_ms': float(plan.expected_cost_ms),
            'policy_expected_waste_ms': float(plan.expected_waste_ms),
            'policy_score_ms': float(plan.score),
            'policy_confidence': float(plan.confidence),
            'policy_reason': plan.reason,
            'admission_controller_enabled': True,
            'native_runtime_family': self.runtime_family,
        }

    def _plan_native_admission(self, tokens: Tuple[int, ...], match, metadata: Dict | None = None) -> ReusePlan:
        exact_len = int(match[2]) if match is not None else 0
        sample_len = max(min(len(tokens), 32), 1)
        full_mpt = self.ewma_full_ms_per_token or max(self.backend.estimate_prefill_cost_ms(sample_len) / sample_len, 0.05)
        return self.controller.plan(
            tokens=tokens,
            exact_match_len=exact_len,
            semantic_similarity=0.0,
            semantic_prefix_len=0,
            shared_prefix_hint=self._shared_prefix_hint(tokens, metadata),
            full_ms_per_token=full_mpt,
            reuse_overhead_ms=self.ewma_cache_reuse_overhead_ms,
            metadata=metadata,
        )

    def _should_store_native_placeholder_with_admission(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> bool:
        prefix_len = self._reactive_prefix_len(tokens, metadata)
        if prefix_len < self.bank.min_match_length:
            self.engine_metrics['admission_store_bypass_total'] = int(self.engine_metrics.get('admission_store_bypass_total', 0)) + 1
            return False
        if self.bank.contains(tokens[:prefix_len]):
            return False

        prefix = tokens[:prefix_len]
        observations = self.bank.get_observation_count(prefix)
        frequency = self.bank.get_frequency(prefix)
        entropy = token_entropy(prefix)
        hint_len = self._shared_prefix_hint(tokens, metadata)
        scaffold = hint_len is not None and prefix_len >= hint_len and self._has_scaffold_hint(tokens, metadata)
        ms_per_token = max(self.ewma_full_ms_per_token or self.backend.estimate_prefill_cost_ms(max(prefix_len, 1)) / max(prefix_len, 1), 0.05)
        expected_benefit = prefix_len * ms_per_token * (1.10 if scaffold else min(0.25 + frequency + 0.10 * observations, 1.0))
        expected_cost = self.ewma_cache_reuse_overhead_ms + (0.02 * max(len(tokens) - prefix_len, 0))
        expected_waste = expected_benefit * min(max(entropy / 16.0, 0.03), 0.35) if observations < 2 and not scaffold else expected_cost * 0.03
        allow = scaffold or (observations >= 2 and expected_benefit - expected_cost - expected_waste >= 0.0)
        self.engine_metrics['admission_store_total' if allow else 'admission_store_bypass_total'] = int(
            self.engine_metrics.get('admission_store_total' if allow else 'admission_store_bypass_total', 0)
        ) + 1
        return allow

    def _store_native_placeholder(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> None:
        if not self._should_store_native_placeholder_with_admission(tokens, metadata):
            return
        super()._store_native_placeholder(tokens, metadata)

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...], metadata: Dict | None = None) -> RequestResult:
        self._observe_request(tokens, metadata)
        match = self.bank.peek_match(tokens)
        plan = self._plan_native_admission(tokens, match, metadata)
        self._record_admission_plan(plan)

        matched_prefix_length = 0
        was_cache_hit = False
        if match is not None and plan.strategy == 'exact':
            _, entry, match_len = match
            if self._should_attempt_cache_use(tokens, entry, match_len, metadata=metadata):
                admitted_entry = self.bank.admit_match(match[0])
                if admitted_entry is not None:
                    matched_prefix_length = match_len
                    was_cache_hit = True
                    self.controller.update_feedback(hit=True, wasted_ratio=0.0)
                else:
                    self.controller.update_feedback(hit=False, wasted_ratio=0.0)
            else:
                self.controller.update_feedback(hit=False, wasted_ratio=0.0)
        else:
            self.controller.update_feedback(hit=False, wasted_ratio=0.0)

        out = self._prefill_full(tokens)
        self._store_native_placeholder(tokens, metadata)
        return self._record(
            RequestResult(
                request_id=request_id,
                latency_ms=out.latency_ms,
                matched_prefix_length=matched_prefix_length,
                tokens_recomputed=max(len(tokens) - matched_prefix_length, 0) if was_cache_hit else len(tokens),
                was_cache_hit=was_cache_hit,
                was_speculative_hit=False,
                gpu_utilization_pct=out.gpu_utilization_pct,
            )
        )

class ReactivePrefixCacheEngine(BaseEngine):
    def __init__(self, backend: Backend, max_memory_mb: int = 256):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb, name='reactive_prefix_cache')

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...], metadata: Dict | None = None) -> RequestResult:
        self._observe_request(tokens, metadata)
        match = self.bank.peek_match(tokens)

        if match is None:
            self.bank.record_miss()
            out = self._prefill_full(tokens)
            store_latency_ms = self._store_reactive_prefix(tokens, tier='cpu', metadata=metadata) if self.cache_enabled else 0.0
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
        if not self._should_attempt_cache_use(tokens, entry, match_len, metadata=metadata):
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

        out, recomputed_tokens, cache_hit, actual_match_len = self._prefill_with_cache_fallback(tokens, admitted_entry, match_len)
        return self._record(
            RequestResult(
                request_id=request_id,
                latency_ms=out.latency_ms,
                matched_prefix_length=actual_match_len if cache_hit else 0,
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

    def _reactive_prefix_len(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> int:
        return self._request_cacheable_prefix_limit(tokens, metadata)

    def _should_store_reactive_prefix(self, tokens: Tuple[int, ...], prefix_len: int, metadata: Dict | None = None) -> bool:
        self.engine_metrics['store_attempts'] += 1
        if not getattr(self.backend, 'supports_external_kv', True):
            self.engine_metrics['store_skips'] += 1
            return False
        if len(tokens) < self.bank.min_match_length:
            self.engine_metrics['store_skips'] += 1
            return False
        return True

    def _should_attempt_cache_use(self, tokens: Tuple[int, ...], entry: CacheEntry, match_len: int, metadata: Dict | None = None) -> bool:
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
        self.policy = FrequencyPolicy(min_frequency=0.22, min_prefix_len=max(self.tuning.min_store_prefix_tokens, 12), max_prefix_len=96, min_observations=2)
        self.speculative_k = speculative_k
        self.idle_threshold_ms = idle_threshold_ms
        self._bootstrap_speculation_observations = int(getattr(self.policy, 'min_observations', 0))
        self.last_request_time = time.time()
        self.serving_event = threading.Event()
        self.stop_event = threading.Event()
        self._background_exception: Exception | None = None
        self.speculative_log: List[Dict] = []
        self.thread: threading.Thread | None = None
        self._speculation_lock = threading.RLock()
        self._active_speculation_started_at: float | None = None
        self._active_overlap_started_at: float | None = None

    def _should_defer_reactive_store(self, tokens: Tuple[int, ...], prefix_len: int, metadata: Dict | None = None) -> bool:
        if not self._has_scaffold_hint(tokens, metadata):
            return False
        if self._bootstrap_speculation_observations <= 0:
            return False
        prefix = tokens[:prefix_len]
        observations = self.bank.get_observation_count(prefix)
        return observations <= self._bootstrap_speculation_observations and not self.bank.contains(prefix)

    def _ensure_worker_started(self) -> None:
        if self.thread is not None or not getattr(self.backend, 'supports_external_kv', True):
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _check_background_exception(self) -> None:
        if self._background_exception is not None:
            raise RuntimeError(f'{self.name} background worker failed') from self._background_exception

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...], metadata: Dict | None = None) -> RequestResult:
        self._check_background_exception()
        self._ensure_worker_started()
        with self._speculation_lock:
            if self._active_speculation_started_at is not None and self._active_overlap_started_at is None:
                self._active_overlap_started_at = time.perf_counter()
        self.serving_event.set()
        self.last_request_time = time.time()
        try:
            return super().serve_tokens(request_id, tokens, metadata=metadata)
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
        try:
            while not self.stop_event.is_set():
                time.sleep(0.005)
                if not self.cache_enabled:
                    continue
                idle_ms = (time.time() - self.last_request_time) * 1000.0
                if self.serving_event.is_set() or idle_ms < self.idle_threshold_ms:
                    continue

                decisions = self.policy.rank(self.bank, budget_k=self.speculative_k, prefer_gpu=False)
                for decision in decisions:
                    if self.serving_event.is_set() or self.stop_event.is_set():
                        break
                    if self.bank.contains(decision.prefix_tokens) or not self._speculation_allowed(decision.prefix_tokens):
                        continue

                    start = time.perf_counter()
                    with self._speculation_lock:
                        self._active_speculation_started_at = start
                        self._active_overlap_started_at = None
                    out = self.backend.prefill(decision.prefix_tokens)
                    self._update_full_cost_stats(len(decision.prefix_tokens), out.latency_ms)
                    cache_obj, move_latency_ms, memory_bytes = self._materialize_cache_for_tier(out.kv_cache, len(decision.prefix_tokens), 'cpu')
                    total_latency_ms = float(out.latency_ms + move_latency_ms)
                    stored = self.bank.store(
                        decision.prefix_tokens,
                        cache_obj,
                        total_latency_ms,
                        memory_bytes,
                        is_speculative=True,
                        tier='cpu',
                    )
                    if not stored:
                        with self._speculation_lock:
                            overlap_started_at = self._active_overlap_started_at
                            self._active_speculation_started_at = None
                            self._active_overlap_started_at = None
                        if overlap_started_at is not None:
                            self.engine_metrics['speculative_overlap_ms'] = float(self.engine_metrics.get('speculative_overlap_ms', 0.0)) + max((time.perf_counter() - overlap_started_at) * 1000.0, 0.0)
                            self.engine_metrics['speculative_overlap_events'] = int(self.engine_metrics.get('speculative_overlap_events', 0)) + 1
                        continue
                    with self._speculation_lock:
                        overlap_started_at = self._active_overlap_started_at
                        self._active_speculation_started_at = None
                        self._active_overlap_started_at = None
                    if overlap_started_at is not None:
                        self.engine_metrics['speculative_overlap_ms'] = float(self.engine_metrics.get('speculative_overlap_ms', 0.0)) + max((time.perf_counter() - overlap_started_at) * 1000.0, 0.0)
                        self.engine_metrics['speculative_overlap_events'] = int(self.engine_metrics.get('speculative_overlap_events', 0)) + 1
                    self.speculative_log.append(
                        {
                            'policy': 'frequency',
                            'score': decision.score,
                            'latency_ms': total_latency_ms,
                            'target_tier': 'cpu',
                        }
                    )
        except Exception as exc:
            self._background_exception = exc
            self.stop_event.set()

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                raise RuntimeError(f'{self.name} background worker did not stop cleanly')
        self._check_background_exception()
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
        self._controller_lock = threading.RLock()
        self.last_request_time = time.time()
        self.serving_event = threading.Event()
        self.stop_event = threading.Event()
        self._background_exception: Exception | None = None
        self.speculative_log: List[Dict] = []
        self._bootstrap_speculation_observations = int(getattr(self.policy, 'min_observations', 0))
        self._min_requests_before_speculation = max(int(getattr(self.policy, 'min_observations', 1)), 1)
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
        self._speculation_lock = threading.RLock()
        self._active_speculation_started_at: float | None = None
        self._active_overlap_started_at: float | None = None
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
        self.thread: threading.Thread | None = None

    def _should_defer_reactive_store(self, tokens: Tuple[int, ...], prefix_len: int, metadata: Dict | None = None) -> bool:
        if not self._has_scaffold_hint(tokens, metadata):
            return False
        if self._bootstrap_speculation_observations <= 0:
            return False
        prefix = tokens[:prefix_len]
        observations = self.bank.get_observation_count(prefix)
        if observations > self._bootstrap_speculation_observations:
            return False
        if self.bank.contains(prefix):
            return False
        return True

    def _ensure_worker_started(self) -> None:
        if self.thread is not None or not getattr(self.backend, 'supports_external_kv', True):
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _check_background_exception(self) -> None:
        if self._background_exception is not None:
            raise RuntimeError(f'{self.name} background worker failed') from self._background_exception

    def _after_record(self, result: RequestResult) -> None:
        total_tokens = max(result.tokens_recomputed + result.matched_prefix_length, 1)
        with self._controller_lock:
            self._recent_request_window.append((total_tokens, result.matched_prefix_length, result.was_cache_hit))

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...], metadata: Dict | None = None) -> RequestResult:
        self._check_background_exception()
        self._ensure_worker_started()
        with self._speculation_lock:
            if self._active_speculation_started_at is not None and self._active_overlap_started_at is None:
                self._active_overlap_started_at = time.perf_counter()
        self.serving_event.set()
        self.last_request_time = time.time()
        try:
            self._observe_request(tokens, metadata)
            match = self.bank.peek_match(tokens)
            if match is None:
                self.bank.record_miss()
                out = self._prefill_full(tokens)
                store_latency_ms = self._store_reactive_prefix(tokens, tier='cpu', metadata=metadata)
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
                if not self._should_attempt_cache_use(tokens, entry, match_len, metadata=metadata):
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
                        out, recomputed_tokens, cache_hit, actual_match_len = self._prefill_with_cache_fallback(tokens, admitted_entry, match_len)
                        promotion_latency_ms = 0.0
                        if (
                            cache_hit
                            and self._should_promote_entry(admitted_entry)
                        ):
                            move_start = time.perf_counter()
                            promoted = self.backend.move_kv_cache(admitted_entry.kv_cache, self.backend.device)
                            promotion_latency_ms = (time.perf_counter() - move_start) * 1000.0
                            promoted_memory_bytes = estimate_past_key_values_bytes(promoted) or self.backend.estimate_kv_cache_bytes(len(admitted_entry.prefix_tokens))
                            if self.bank.promote(admitted_entry.prefix_tokens, promoted, promoted_memory_bytes, new_tier='gpu'):
                                admitted_entry.tier = 'gpu'

                        result = RequestResult(
                            request_id=request_id,
                            latency_ms=out.latency_ms + promotion_latency_ms,
                            matched_prefix_length=actual_match_len if cache_hit else 0,
                            tokens_recomputed=recomputed_tokens,
                            was_cache_hit=cache_hit,
                            was_speculative_hit=admitted_entry.was_speculative if cache_hit else False,
                            cache_tier=admitted_entry.tier if cache_hit else None,
                            gpu_utilization_pct=out.gpu_utilization_pct,
                        )

            return self._record(result)
        finally:
            self.serving_event.clear()

    def _should_promote_entry(self, entry: CacheEntry) -> bool:
        if not self.enable_gpu_tier or entry.tier != 'cpu':
            return False
        prefix_len = len(entry.prefix_tokens)
        if prefix_len > self.tuning.gpu_promotion_max_prefix_tokens:
            return False
        promote_hits = self.tuning.promote_after_hits
        if entry.was_speculative:
            promote_hits = max(promote_hits - 1, 1)
        if entry.generation_cost_ms >= max(self.tuning.min_estimated_saved_ms * 4.0, 16.0):
            promote_hits = max(promote_hits - 1, 1)
        return entry.hit_count >= promote_hits

    def _speculation_allowed(self, decision_prefix: Tuple[int, ...], decision_benefit_ms: float, decision_cost_ms: float) -> bool:
        if len(decision_prefix) < self.tuning.min_store_prefix_tokens:
            return False
        estimated_cost = self._estimate_full_cost_ms(len(decision_prefix))
        recent_support = self.bank.recent_prefix_support(decision_prefix)
        recent_streak = self.bank.recent_prefix_streak(decision_prefix)
        if recent_support <= 0.0 and recent_streak == 0:
            return False
        long_scaffold_candidate = len(decision_prefix) >= max(self.tuning.min_store_prefix_tokens * 4, 48)
        if self._cpu_mode:
            support_gate = recent_support >= 0.08 or recent_streak >= 2
        else:
            support_gate = recent_support >= 0.10 or recent_streak >= 2
        if long_scaffold_candidate:
            support_gate = support_gate or recent_streak >= 1 or recent_support >= (0.04 if self._cpu_mode else 0.03)
        required_benefit = decision_cost_ms * (0.85 if long_scaffold_candidate else 1.0)
        required_estimated_cost = self.tuning.min_estimated_saved_ms * (0.5 if long_scaffold_candidate else 1.0)
        return (
            decision_benefit_ms >= max(required_benefit, 0.0)
            and estimated_cost >= required_estimated_cost
            and support_gate
        )

    def _refresh_speculation_controller(self) -> bool:
        now = time.time()
        snapshot = self.bank.snapshot_metrics()
        with self._controller_lock:
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
            meaningful_absolute_reuse = self._has_meaningful_absolute_reuse(recent_reused_tokens, recent_hits)
            self.engine_metrics['recent_reuse_density_final'] = float(recent_reuse_density)
            self.engine_metrics['recent_hit_rate_final'] = float(recent_hit_rate)

            if requests_seen < self._min_requests_before_speculation:
                self._effective_speculative_k = 0
            elif now < self._speculation_cooldown_until:
                self._effective_speculative_k = 0
            elif pending_speculative >= self._max_pending_speculative:
                self._effective_speculative_k = 0
            elif recent_request_count >= 6 and recent_reuse_density < 0.04 and recent_hit_rate < 0.10 and not meaningful_absolute_reuse:
                self._effective_speculative_k = 0
            elif recent_count >= 2 and recent_net <= 0.0:
                self._effective_speculative_k = 0
            elif recent_count >= 1 and recent_net < (6.0 if meaningful_absolute_reuse else 12.0):
                self._effective_speculative_k = 1
            elif self._cpu_mode and (recent_hit_rate >= 0.30 or recent_reuse_density >= 0.10):
                self._effective_speculative_k = min(max(self.speculative_k, 1), 2)
            elif (not self._cpu_mode) and (recent_hit_rate >= 0.10 or recent_reuse_density >= 0.04 or meaningful_absolute_reuse):
                self._effective_speculative_k = min(max(self.speculative_k, 1), 2)
            elif recent_hit_rate < 0.20 and recent_reuse_density < 0.08 and not meaningful_absolute_reuse:
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
        try:
            while not self.stop_event.is_set():
                time.sleep(0.005)
                if not self.cache_enabled:
                    continue
                idle_ms = (time.time() - self.last_request_time) * 1000.0
                if self.serving_event.is_set() or idle_ms < self.idle_threshold_ms:
                    continue
                if not self._refresh_speculation_controller():
                    continue

                prefer_gpu = self.enable_gpu_tier and idle_ms >= self.idle_threshold_ms
                with self._controller_lock:
                    budget_k = self._effective_speculative_k
                decisions = self.policy.rank(self.bank, budget_k=budget_k, prefer_gpu=prefer_gpu)

                for decision in decisions:
                    if self.serving_event.is_set() or self.stop_event.is_set():
                        break
                    if self.bank.contains(decision.prefix_tokens):
                        continue
                    if not self._speculation_allowed(decision.prefix_tokens, decision.expected_benefit_ms, decision.expected_cost_ms):
                        continue

                    start = time.perf_counter()
                    with self._speculation_lock:
                        self._active_speculation_started_at = start
                        self._active_overlap_started_at = None
                    out = self.backend.prefill(decision.prefix_tokens)
                    self._update_full_cost_stats(len(decision.prefix_tokens), out.latency_ms)
                    recent_streak = self.bank.recent_prefix_streak(decision.prefix_tokens)
                    target_tier = 'cpu'
                    if self.enable_gpu_tier and (recent_streak >= 1 or decision.expected_benefit_ms >= (decision.expected_cost_ms * 1.05)):
                        target_tier = 'gpu'
                    cache_obj, move_latency_ms, memory_bytes = self._materialize_cache_for_tier(out.kv_cache, len(decision.prefix_tokens), target_tier)
                    total_latency_ms = float(out.latency_ms + move_latency_ms)

                    stored = self.bank.store(
                        decision.prefix_tokens,
                        cache_obj,
                        total_latency_ms,
                        memory_bytes,
                        is_speculative=True,
                        tier=target_tier,
                    )
                    if not stored:
                        with self._speculation_lock:
                            overlap_started_at = self._active_overlap_started_at
                            self._active_speculation_started_at = None
                            self._active_overlap_started_at = None
                        if overlap_started_at is not None:
                            self.engine_metrics['speculative_overlap_ms'] = float(self.engine_metrics.get('speculative_overlap_ms', 0.0)) + max((time.perf_counter() - overlap_started_at) * 1000.0, 0.0)
                            self.engine_metrics['speculative_overlap_events'] = int(self.engine_metrics.get('speculative_overlap_events', 0)) + 1
                        continue
                    with self._speculation_lock:
                        overlap_started_at = self._active_overlap_started_at
                        self._active_speculation_started_at = None
                        self._active_overlap_started_at = None
                    if overlap_started_at is not None:
                        self.engine_metrics['speculative_overlap_ms'] = float(self.engine_metrics.get('speculative_overlap_ms', 0.0)) + max((time.perf_counter() - overlap_started_at) * 1000.0, 0.0)
                        self.engine_metrics['speculative_overlap_events'] = int(self.engine_metrics.get('speculative_overlap_events', 0)) + 1
                    self.speculative_log.append(
                        {
                            'policy': 'template_aware_cost_aware_slack',
                            'score': decision.score,
                            'latency_ms': total_latency_ms,
                            'target_tier': target_tier,
                            'expected_benefit_ms': decision.expected_benefit_ms,
                            'expected_cost_ms': decision.expected_cost_ms,
                        }
                    )
        except Exception as exc:
            self._background_exception = exc
            self.stop_event.set()

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                raise RuntimeError(f'{self.name} background worker did not stop cleanly')
        self._check_background_exception()
        self.finalize()



class ShadowKVPlusEngine(ShadowKVEngine):
    """ShadowKV++: policy-driven, semantic, fine-grained KV reuse.

    Real backends only reuse exact-prefix KV for correctness. Semantic retrieval
    drives the controller and can simulate approximate partial reuse on the
    FakeBackend so the research code can measure the opportunity without adding
    a heavy embedding model dependency.
    """

    def __init__(
        self,
        backend: Backend,
        max_memory_mb: int = 256,
        policy: SpeculationPolicy | None = None,
        speculative_k: int = 4,
        idle_threshold_ms: float = 60.0,
        enable_gpu_tier: bool = True,
        semantic_similarity_threshold: float = 0.58,
        allow_approximate_semantic_reuse: bool | None = None,
        semantic_ablation_mode: str = 'safe',
        raw_strategy: str | None = None,
        early_layer_reuse_ratio: float = 0.35,
        logit_guard_threshold: float = 0.08,
    ):
        super().__init__(backend=backend, max_memory_mb=max_memory_mb, policy=policy, speculative_k=speculative_k, idle_threshold_ms=idle_threshold_ms, enable_gpu_tier=enable_gpu_tier)
        self.semantic_ablation_mode = str(semantic_ablation_mode or 'safe')
        self._semantic_execution_modes = {'scaffold_only', 'early_layer', 'logit_guard'}
        self.raw_strategy = str(raw_strategy or ('fastpath' if self.semantic_ablation_mode == 'best_latency' else 'strict_utility_gate'))
        self.name = 'shadow_kv_plus' if self.semantic_ablation_mode == 'safe' else f'shadow_kv_plus_{self.semantic_ablation_mode}'
        self.fast_raw_bypass_enabled = True
        self.fast_raw_bypass_min_requests = 8
        self.fast_raw_bypass_density_threshold = 0.04
        self.fast_raw_bypass_success_threshold = 1
        # Raw prompts are the danger zone. Main ShadowKV++ now starts in a
        # strict no-store/no-spec gate and graduates only when long, repeated
        # prefixes show positive expected utility. The previous best-latency
        # fastpath is preserved as shadow_kv_plus_best_latency, and a raw
        # observer/no-store baseline is available as shadow_kv_plus_raw_observer.
        self.raw_conservative_start_enabled = self.raw_strategy in {'strict_utility_gate', 'raw_observer'}
        self.raw_graduate_min_observations = 8
        self.raw_graduate_min_frequency = 0.35
        self.raw_graduate_min_prefix_len = 48
        self.raw_graduate_min_requests = 12
        self.raw_graduate_min_estimated_net_ms = 8.0
        self.semantic_index = SemanticKVIndex(dims=128, max_entries=1024)
        self.controller = AdaptiveReuseController(min_utility_ms=0.0, semantic_threshold=semantic_similarity_threshold, max_layer_reuse_ratio=0.55)
        self.allow_approximate_semantic_reuse = (getattr(backend, 'backend_name', '') == 'fake') if allow_approximate_semantic_reuse is None else bool(allow_approximate_semantic_reuse)
        self.early_layer_reuse_ratio = max(0.05, min(float(early_layer_reuse_ratio), 1.0))
        self.logit_guard_threshold = max(float(logit_guard_threshold), 0.0)
        self.engine_metrics.update({
            'semantic_ablation_mode': self.semantic_ablation_mode,
            'policy_plans_total': 0, 'policy_bypass_total': 0, 'policy_exact_total': 0, 'policy_semantic_partial_total': 0,
            'semantic_index_entries_final': 0, 'semantic_queries_total': 0, 'semantic_matches_total': 0,
            'semantic_partial_hits': 0, 'semantic_partial_reused_tokens_total': 0,
            'semantic_opportunity_plans_total': 0, 'semantic_opportunity_reused_tokens_total': 0,
            'semantic_opportunity_estimated_savings_ms': 0.0,
            'semantic_blocked_by_backend_total': 0,
            'fast_raw_bypass_total': 0,
            'raw_strategy': self.raw_strategy,
            'raw_conservative_bypass_total': 0,
            'raw_observer_bypass_total': 0,
            'raw_reuse_evidence_strong_total': 0,
            'raw_reuse_evidence_weak_total': 0,
            'raw_graduated_total': 0,
            'raw_graduate_min_observations': self.raw_graduate_min_observations,
            'raw_graduate_min_frequency': self.raw_graduate_min_frequency,
            'raw_graduate_min_prefix_len': self.raw_graduate_min_prefix_len,
            'semantic_queries_skipped_total': 0,
            'scaffold_only_attempts': 0, 'scaffold_only_hits': 0,
            'early_layer_attempts': 0, 'early_layer_hits': 0,
            'early_layer_reuse_ratio_sum': 0.0,
            'logit_guard_checks': 0, 'logit_guard_passes': 0, 'logit_guard_failures': 0,
            'logit_guard_distance_sum': 0.0, 'logit_guard_threshold': self.logit_guard_threshold,
            'semantic_guarded_hits': 0,
            'semantic_quality_divergence_sum': 0.0, 'semantic_quality_divergence_events': 0,
            'layer_reuse_ratio_sum': 0.0, 'layer_reuse_events': 0,
            'policy_expected_benefit_ms': 0.0, 'policy_expected_cost_ms': 0.0, 'policy_expected_waste_ms': 0.0, 'policy_net_utility_ms': 0.0,
        })

    def _semantic_enabled_for_request(self, metadata: Dict | None = None) -> bool:
        prompt_mode = str((metadata or {}).get('prompt_mode', '')).strip().lower()
        return prompt_mode == 'semantic' or self.semantic_ablation_mode in self._semantic_execution_modes

    def _observe_request(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> None:
        super()._observe_request(tokens, metadata)
        # Semantic indexing is useful only for semantic/paraphrase evaluation or
        # explicit semantic ablations. Avoid this overhead on raw/templated
        # performance runs, where exact prefix matching is sufficient.
        if not self._semantic_enabled_for_request(metadata):
            return
        for length in self._tracked_prefix_lengths(tokens, metadata):
            if length >= self.bank.min_match_length:
                self.semantic_index.add(tokens[:length], semantic_key=str((metadata or {}).get('semantic_equivalence_key') or ''))
        self.engine_metrics['semantic_index_entries_final'] = len(self.semantic_index._rows)

    def _semantic_best_match(self, tokens: Tuple[int, ...], metadata: Dict | None = None):
        meta = metadata or {}
        if not self._semantic_enabled_for_request(meta):
            self.engine_metrics['semantic_queries_skipped_total'] = int(self.engine_metrics.get('semantic_queries_skipped_total', 0)) + 1
            return None, 0.0, 0, 0
        self.engine_metrics['semantic_queries_total'] = int(self.engine_metrics.get('semantic_queries_total', 0)) + 1
        semantic_key = str(meta.get('semantic_equivalence_key') or '') or None
        matches = self.semantic_index.query(tokens, k=8, min_similarity=0.20, semantic_key=semantic_key)
        if matches:
            self.engine_metrics['semantic_matches_total'] = int(self.engine_metrics.get('semantic_matches_total', 0)) + len(matches)
        best = None
        current_hint = self._shared_prefix_hint(tokens, meta)
        current_prefix = tokens[:current_hint] if current_hint else None
        prompt_mode = str(meta.get('prompt_mode', '')).lower()
        for m in matches:
            # Never count the current just-observed scaffold as its own semantic
            # neighbour unless it is already cached. Otherwise semantic runs are
            # mislabeled as exact self-matches and opportunity metrics stay zero.
            if current_prefix is not None and m.prefix_tokens == current_prefix and not self.bank.contains(m.prefix_tokens):
                continue
            if tokens[: len(m.prefix_tokens)] == m.prefix_tokens and not self.bank.contains(m.prefix_tokens):
                continue
            lcp = longest_common_prefix_len(tokens, m.prefix_tokens)
            same_family_bonus = 0.35 if semantic_key else 0.0
            diversity_bonus = 0.15 if (prompt_mode == 'semantic' and current_prefix is not None and m.prefix_tokens != current_prefix) else 0.0
            score = m.similarity + same_family_bonus + diversity_bonus + 0.10 * min(lcp / max(len(m.prefix_tokens), 1), 1.0)
            if best is None or score > best[0]:
                best = (score, m.prefix_tokens, float(m.similarity), int(m.prefix_len), int(lcp))
        if best is None:
            return None, 0.0, 0, 0
        return best[1], best[2], best[3], best[4]

    def _record_plan(self, plan: ReusePlan) -> None:
        self.engine_metrics['policy_plans_total'] = int(self.engine_metrics.get('policy_plans_total', 0)) + 1
        key = 'policy_bypass_total' if plan.strategy == 'bypass' else ('policy_exact_total' if plan.strategy == 'exact' else 'policy_semantic_partial_total')
        self.engine_metrics[key] = int(self.engine_metrics.get(key, 0)) + 1
        self.engine_metrics['layer_reuse_ratio_sum'] = float(self.engine_metrics.get('layer_reuse_ratio_sum', 0.0)) + float(plan.layer_reuse_ratio)
        if plan.layer_reuse_ratio > 0:
            self.engine_metrics['layer_reuse_events'] = int(self.engine_metrics.get('layer_reuse_events', 0)) + 1
        self.engine_metrics['policy_expected_benefit_ms'] = float(self.engine_metrics.get('policy_expected_benefit_ms', 0.0)) + plan.expected_benefit_ms
        self.engine_metrics['policy_expected_cost_ms'] = float(self.engine_metrics.get('policy_expected_cost_ms', 0.0)) + plan.expected_cost_ms
        self.engine_metrics['policy_expected_waste_ms'] = float(self.engine_metrics.get('policy_expected_waste_ms', 0.0)) + plan.expected_waste_ms
        self.engine_metrics['policy_net_utility_ms'] = float(self.engine_metrics.get('policy_net_utility_ms', 0.0)) + plan.score

    def _plan_for_request(self, tokens: Tuple[int, ...], match, metadata: Dict | None = None):
        exact_len = int(match[2]) if match is not None else 0
        semantic_key, sem_sim, sem_prefix_len, sem_lcp = self._semantic_best_match(tokens, metadata)
        self._current_semantic_match = {
            'semantic_similarity': float(sem_sim),
            'semantic_prefix_len': int(sem_prefix_len),
            'semantic_lcp_len': int(sem_lcp),
            'semantic_match_available': bool(semantic_key is not None),
        }
        # A semantic neighbour is not an exact KV prefix unless the bank
        # independently found an exact cache match. Do not relabel semantic
        # opportunities as exact matches in semantic prompt mode.
        prompt_mode = str((metadata or {}).get('prompt_mode', '')).lower()
        if prompt_mode != 'semantic' and exact_len <= 0 and sem_lcp >= self.tuning.min_reuse_prefix_tokens:
            exact_len = sem_lcp
        sample_len = max(min(len(tokens), 32), 1)
        full_mpt = self.ewma_full_ms_per_token or max(self.backend.estimate_prefill_cost_ms(sample_len) / sample_len, 0.05)
        plan = self.controller.plan(tokens=tokens, exact_match_len=exact_len, semantic_similarity=sem_sim, semantic_prefix_len=sem_prefix_len, shared_prefix_hint=self._shared_prefix_hint(tokens, metadata), full_ms_per_token=full_mpt, reuse_overhead_ms=self.ewma_cache_reuse_overhead_ms, metadata=metadata)
        self._current_trace_plan = {
            'policy_strategy': plan.strategy,
            'semantic_ablation_mode': self.semantic_ablation_mode,
            'policy_speculate_depth_tokens': int(plan.speculate_depth_tokens),
            'policy_reusable_prefix_tokens': int(plan.reusable_prefix_tokens),
            'policy_layer_reuse_ratio': float(plan.layer_reuse_ratio),
            'policy_expected_benefit_ms': float(plan.expected_benefit_ms),
            'policy_expected_cost_ms': float(plan.expected_cost_ms),
            'policy_expected_waste_ms': float(plan.expected_waste_ms),
            'policy_score_ms': float(plan.score),
            'policy_confidence': float(plan.confidence),
            'policy_reason': plan.reason,
        }
        self._record_plan(plan)
        return plan, semantic_key

    def _should_defer_reactive_store(self, tokens: Tuple[int, ...], prefix_len: int, metadata: Dict | None = None) -> bool:
        # For raw, low-reuse workloads, ShadowKV++ should not pay repeated
        # prefill/store overhead once early evidence says reuse density is low.
        # This is deliberately scoped to raw mode so templated/semantic scaffolds
        # still get aggressive cold-start materialization.
        if self._is_raw_fast_bypass_active(metadata):
            return True
        # ShadowKV++ front-loads long, explicit scaffolds because the controller
        # has high confidence they will be reused; this fixes the cold-start
        # weakness of idle-only speculation on templated serving workloads.
        hint = self._shared_prefix_hint(tokens, metadata)
        if hint is not None and hint >= max(self.tuning.min_store_prefix_tokens * 4, 24):
            return False
        return super()._should_defer_reactive_store(tokens, prefix_len, metadata)

    def _store_reactive_prefix(self, tokens: Tuple[int, ...], tier: str = 'cpu', metadata: Dict | None = None) -> float:
        latency = super()._store_reactive_prefix(tokens, tier=tier, metadata=metadata)
        prefix_len = self._reactive_prefix_len(tokens, metadata)
        if prefix_len >= self.bank.min_match_length:
            self.semantic_index.add(tokens[:prefix_len], semantic_key=str((metadata or {}).get('semantic_equivalence_key') or ''))
            self.engine_metrics['semantic_index_entries_final'] = len(self.semantic_index._rows)
        return latency

    def _record_semantic_opportunity(self, plan: ReusePlan, semantic_key: Tuple[int, ...] | None) -> None:
        if plan.strategy != 'semantic_partial' or semantic_key is None:
            return
        reusable = min(plan.reusable_prefix_tokens, len(semantic_key))
        if reusable < self.bank.min_match_length:
            return
        self.engine_metrics['semantic_opportunity_plans_total'] = int(self.engine_metrics.get('semantic_opportunity_plans_total', 0)) + 1
        self.engine_metrics['semantic_opportunity_reused_tokens_total'] = int(self.engine_metrics.get('semantic_opportunity_reused_tokens_total', 0)) + reusable
        self.engine_metrics['semantic_opportunity_estimated_savings_ms'] = float(self.engine_metrics.get('semantic_opportunity_estimated_savings_ms', 0.0)) + max(plan.expected_benefit_ms - plan.expected_cost_ms, 0.0)
        if not self.allow_approximate_semantic_reuse and self.semantic_ablation_mode == 'safe':
            self.engine_metrics['semantic_blocked_by_backend_total'] = int(self.engine_metrics.get('semantic_blocked_by_backend_total', 0)) + 1

    def _store_semantic_scaffold_prefix(self, tokens: Tuple[int, ...], metadata: Dict | None = None, tier: str = 'cpu') -> float:
        """Materialize the current request scaffold for semantic ablations.

        This is deliberately separated from the default safe path. It creates
        concrete KV candidates for scaffold-only, early-layer, and guarded-reuse
        ablations, at an explicit measured cost.
        """
        hint = self._shared_prefix_hint(tokens, metadata)
        if hint is None or hint < self.bank.min_match_length:
            return 0.0
        prefix = tokens[: min(hint, self.tuning.max_cacheable_prefix_tokens)]
        if self.bank.contains(prefix):
            return 0.0
        if not getattr(self.backend, 'supports_external_kv', True):
            return 0.0
        out = self.backend.prefill(prefix)
        self._update_full_cost_stats(len(prefix), out.latency_ms)
        cache_obj, move_latency_ms, memory_bytes = self._materialize_cache_for_tier(out.kv_cache, len(prefix), tier)
        total_latency_ms = float(out.latency_ms + move_latency_ms)
        if self.bank.store(prefix, cache_obj, total_latency_ms, memory_bytes, is_speculative=False, tier=tier):
            self.engine_metrics['store_successes'] = int(self.engine_metrics.get('store_successes', 0)) + 1
            self.engine_metrics['store_latency_total_ms'] = float(self.engine_metrics.get('store_latency_total_ms', 0.0)) + total_latency_ms
            self.semantic_index.add(prefix, semantic_key=str((metadata or {}).get('semantic_equivalence_key') or ''))
            self.engine_metrics['semantic_index_entries_final'] = len(self.semantic_index._rows)
            return total_latency_ms
        return 0.0

    def _quality_divergence_proxy(self, semantic_similarity: float, layer_ratio: float) -> float:
        # A conservative proxy for the speed-quality tradeoff curve: lower
        # semantic similarity and deeper reuse imply higher output divergence.
        return max(0.0, 1.0 - semantic_similarity) * max(layer_ratio, 0.0)

    def _logit_guard_allows(self, current_tokens: Tuple[int, ...], semantic_tokens: Tuple[int, ...], plan: ReusePlan) -> bool:
        self.engine_metrics['logit_guard_checks'] = int(self.engine_metrics.get('logit_guard_checks', 0)) + 1
        probe_len = min(max(plan.reusable_prefix_tokens, self.bank.min_match_length), len(current_tokens), len(semantic_tokens))
        distance = self.backend.logit_guard_distance(current_tokens[:probe_len], semantic_tokens[:probe_len], top_k=32)
        if distance is None:
            self.engine_metrics['logit_guard_failures'] = int(self.engine_metrics.get('logit_guard_failures', 0)) + 1
            if self._current_trace_plan is not None:
                self._current_trace_plan['logit_guard_distance'] = None
                self._current_trace_plan['logit_guard_passed'] = False
            return False
        self.engine_metrics['logit_guard_distance_sum'] = float(self.engine_metrics.get('logit_guard_distance_sum', 0.0)) + float(distance)
        passed = float(distance) <= self.logit_guard_threshold
        self.engine_metrics['logit_guard_passes' if passed else 'logit_guard_failures'] = int(self.engine_metrics.get('logit_guard_passes' if passed else 'logit_guard_failures', 0)) + 1
        if self._current_trace_plan is not None:
            self._current_trace_plan['logit_guard_distance'] = float(distance)
            self._current_trace_plan['logit_guard_passed'] = bool(passed)
        return passed

    def _partial_semantic_reuse(self, request_id: int, tokens: Tuple[int, ...], semantic_key: Tuple[int, ...], plan: ReusePlan) -> RequestResult | None:
        entry = self.bank.get_entry(semantic_key)
        if entry is None:
            return None

        mode = self.semantic_ablation_mode
        if mode == 'scaffold_only':
            self.engine_metrics['scaffold_only_attempts'] = int(self.engine_metrics.get('scaffold_only_attempts', 0)) + 1
        elif mode == 'early_layer':
            self.engine_metrics['early_layer_attempts'] = int(self.engine_metrics.get('early_layer_attempts', 0)) + 1

        if mode == 'scaffold_only':
            # Conservative baseline: execute only if the semantic candidate is
            # also an exact current-token scaffold. Otherwise it remains an
            # opportunity, not an approximate KV substitution.
            hint = self._shared_prefix_hint(tokens, (self._current_trace_context or {}).get('metadata') or {})
            if hint is None or tokens[:min(hint, len(semantic_key))] != semantic_key[:min(hint, len(semantic_key))]:
                return None
            reusable = min(hint, len(semantic_key), len(tokens))
        elif mode == 'early_layer':
            reusable = min(int(plan.reusable_prefix_tokens * self.early_layer_reuse_ratio), len(semantic_key), len(tokens))
            self.engine_metrics['early_layer_reuse_ratio_sum'] = float(self.engine_metrics.get('early_layer_reuse_ratio_sum', 0.0)) + self.early_layer_reuse_ratio
            if self._current_trace_plan is not None:
                self._current_trace_plan['early_layer_reuse_ratio'] = self.early_layer_reuse_ratio
        elif mode == 'logit_guard':
            if not self._logit_guard_allows(tokens, semantic_key, plan):
                return None
            reusable = min(plan.reusable_prefix_tokens, len(semantic_key), len(tokens))
        else:
            if not self.allow_approximate_semantic_reuse:
                return None
            reusable = min(plan.reusable_prefix_tokens, len(semantic_key), len(tokens))

        if reusable < self.bank.min_match_length:
            return None
        if mode not in {'logit_guard'} and not self.allow_approximate_semantic_reuse and getattr(self.backend, 'backend_name', '') != 'fake':
            return None

        out = self.backend.prefill(tokens[reusable:], past_key_values=entry.kv_cache)
        semantic_similarity = float((self._current_semantic_match or {}).get('semantic_similarity') or 0.0)
        divergence = self._quality_divergence_proxy(semantic_similarity, plan.layer_reuse_ratio if mode != 'early_layer' else self.early_layer_reuse_ratio)
        self.engine_metrics['semantic_quality_divergence_sum'] = float(self.engine_metrics.get('semantic_quality_divergence_sum', 0.0)) + divergence
        self.engine_metrics['semantic_quality_divergence_events'] = int(self.engine_metrics.get('semantic_quality_divergence_events', 0)) + 1
        self.engine_metrics['semantic_partial_hits'] = int(self.engine_metrics.get('semantic_partial_hits', 0)) + 1
        self.engine_metrics['semantic_partial_reused_tokens_total'] = int(self.engine_metrics.get('semantic_partial_reused_tokens_total', 0)) + reusable
        if mode == 'scaffold_only':
            self.engine_metrics['scaffold_only_hits'] = int(self.engine_metrics.get('scaffold_only_hits', 0)) + 1
        if mode == 'early_layer':
            self.engine_metrics['early_layer_hits'] = int(self.engine_metrics.get('early_layer_hits', 0)) + 1
        if mode == 'logit_guard':
            self.engine_metrics['semantic_guarded_hits'] = int(self.engine_metrics.get('semantic_guarded_hits', 0)) + 1
        self.engine_metrics['reuse_successes'] = int(self.engine_metrics.get('reuse_successes', 0)) + 1
        if self._current_trace_plan is not None:
            self._current_trace_plan['semantic_quality_divergence_proxy'] = divergence
        self._update_reuse_cost_stats(len(tokens) - reusable, out.latency_ms, reusable)
        self.controller.update_feedback(hit=True, wasted_ratio=0.0)
        return RequestResult(request_id, out.latency_ms, reusable, len(tokens) - reusable, True, entry.was_speculative, cache_tier=entry.tier, gpu_utilization_pct=out.gpu_utilization_pct)

    def _is_raw_prompt(self, metadata: Dict | None = None) -> bool:
        return str((metadata or {}).get('prompt_mode', '')).strip().lower() == 'raw'

    def _raw_reuse_evidence_strong(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> bool:
        """Strict raw-mode graduation gate.

        The earlier raw conservative gate graduated on short repeated scaffolds,
        which caused weak raw datasets to pay cache/store overhead. This gate is
        intentionally harder to pass: it requires a minimum warm-up, a long
        repeated prefix, sufficient observation support, and a positive estimated
        utility margin before raw requests can use ShadowKV++ cache paths.
        """
        if not self._is_raw_prompt(metadata):
            return True
        if self.raw_strategy == 'best_latency':
            self.engine_metrics['raw_reuse_evidence_strong_total'] = int(self.engine_metrics.get('raw_reuse_evidence_strong_total', 0)) + 1
            return True
        if self.raw_strategy == 'raw_observer':
            self.engine_metrics['raw_reuse_evidence_weak_total'] = int(self.engine_metrics.get('raw_reuse_evidence_weak_total', 0)) + 1
            return False

        requests_seen = int(self.engine_metrics.get('requests_seen', 0))
        if requests_seen < self.raw_graduate_min_requests:
            self.engine_metrics['raw_reuse_evidence_weak_total'] = int(self.engine_metrics.get('raw_reuse_evidence_weak_total', 0)) + 1
            return False

        existing = self.bank.peek_match(tokens)
        if existing is not None and existing[2] >= self.raw_graduate_min_prefix_len:
            self.engine_metrics['raw_reuse_evidence_strong_total'] = int(self.engine_metrics.get('raw_reuse_evidence_strong_total', 0)) + 1
            self.engine_metrics['raw_graduated_total'] = int(self.engine_metrics.get('raw_graduated_total', 0)) + 1
            return True

        best_obs = 0
        best_freq = 0.0
        best_len = 0
        for length in self._tracked_prefix_lengths(tokens, metadata):
            if length < self.raw_graduate_min_prefix_len or length > len(tokens):
                continue
            prefix = tokens[:length]
            obs = self.bank.get_observation_count(prefix)
            freq = self.bank.get_frequency(prefix)
            if obs > best_obs or (obs == best_obs and (freq > best_freq or length > best_len)):
                best_obs, best_freq, best_len = obs, freq, length

        ms_per_token = max(float(self.ewma_full_ms_per_token or getattr(self.backend, 'default_ms_per_token', 0.6)), 0.05)
        overhead = max(float(self.ewma_cache_reuse_overhead_ms or getattr(self.backend, 'default_cache_reuse_overhead_ms', 2.0)), 0.0)
        estimated_net_ms = best_len * ms_per_token - overhead
        strong = (
            best_obs >= self.raw_graduate_min_observations
            and best_freq >= self.raw_graduate_min_frequency
            and best_len >= self.raw_graduate_min_prefix_len
            and estimated_net_ms >= self.raw_graduate_min_estimated_net_ms
        )
        if strong:
            self.engine_metrics['raw_reuse_evidence_strong_total'] = int(self.engine_metrics.get('raw_reuse_evidence_strong_total', 0)) + 1
            self.engine_metrics['raw_graduated_total'] = int(self.engine_metrics.get('raw_graduated_total', 0)) + 1
        else:
            self.engine_metrics['raw_reuse_evidence_weak_total'] = int(self.engine_metrics.get('raw_reuse_evidence_weak_total', 0)) + 1
        return strong

    def _is_raw_conservative_bypass_active(self, tokens: Tuple[int, ...], metadata: Dict | None = None) -> bool:
        if not self.raw_conservative_start_enabled or self.semantic_ablation_mode not in {'safe', 'raw_observer'}:
            return False
        if not self._is_raw_prompt(metadata):
            return False
        return not self._raw_reuse_evidence_strong(tokens, metadata)

    def _is_raw_fast_bypass_active(self, metadata: Dict | None = None) -> bool:
        if not self.fast_raw_bypass_enabled or self.semantic_ablation_mode not in {'safe', 'best_latency'}:
            return False
        prompt_mode = str((metadata or {}).get('prompt_mode', '')).strip().lower()
        if prompt_mode != 'raw':
            return False
        requests_seen = int(self.engine_metrics.get('requests_seen', 0))
        if requests_seen < self.fast_raw_bypass_min_requests:
            return False
        reused = int(self.engine_metrics.get('reused_prefix_tokens_total', 0))
        recompute = int(self.engine_metrics.get('recompute_tokens_total', 0))
        successes = int(self.engine_metrics.get('reuse_successes', 0))
        density = reused / max(reused + recompute, 1)
        return density < self.fast_raw_bypass_density_threshold and successes <= self.fast_raw_bypass_success_threshold

    def _fast_bypass_request(self, request_id: int, tokens: Tuple[int, ...]) -> RequestResult:
        self.engine_metrics['fast_raw_bypass_total'] = int(self.engine_metrics.get('fast_raw_bypass_total', 0)) + 1
        self.bank.record_miss()
        out = self._prefill_full(tokens)
        self.controller.update_feedback(hit=False, wasted_ratio=0.0)
        return self._record(RequestResult(request_id, out.latency_ms, 0, len(tokens), False, False, gpu_utilization_pct=out.gpu_utilization_pct))

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...], metadata: Dict | None = None) -> RequestResult:
        self._check_background_exception(); self._ensure_worker_started()
        with self._speculation_lock:
            if self._active_speculation_started_at is not None and self._active_overlap_started_at is None:
                self._active_overlap_started_at = time.perf_counter()
        self.serving_event.set(); self.last_request_time = time.time()
        try:
            self._observe_request(tokens, metadata)
            if self._is_raw_conservative_bypass_active(tokens, metadata):
                self.engine_metrics['raw_conservative_bypass_total'] = int(self.engine_metrics.get('raw_conservative_bypass_total', 0)) + 1
                if self.raw_strategy == 'raw_observer':
                    self.engine_metrics['raw_observer_bypass_total'] = int(self.engine_metrics.get('raw_observer_bypass_total', 0)) + 1
                self._speculation_cooldown_until = max(getattr(self, '_speculation_cooldown_until', 0.0), time.time() + max(self.idle_threshold_ms / 1000.0, 0.25))
                return self._fast_bypass_request(request_id, tokens)
            if self._is_raw_fast_bypass_active(metadata):
                return self._fast_bypass_request(request_id, tokens)
            match = self.bank.peek_match(tokens)
            plan, semantic_key = self._plan_for_request(tokens, match, metadata)
            self._record_semantic_opportunity(plan, semantic_key)
            if plan.strategy == 'bypass':
                self.bank.record_miss()
                out = self._prefill_full(tokens)
                store_latency_ms = 0.0
                if self.semantic_ablation_mode in {'scaffold_only', 'early_layer', 'logit_guard'}:
                    store_latency_ms = self._store_semantic_scaffold_prefix(tokens, metadata=metadata, tier='cpu')
                # Bypass means the policy found no positive utility; do not add
                # reactive store cost on the same request unless an explicit
                # semantic ablation needs scaffold materialization.
                self.controller.update_feedback(hit=False, wasted_ratio=0.0)
                return self._record(RequestResult(request_id, out.latency_ms + store_latency_ms, 0, len(tokens), False, False, gpu_utilization_pct=out.gpu_utilization_pct))
            if match is None and plan.strategy == 'semantic_partial' and semantic_key is not None:
                partial = self._partial_semantic_reuse(request_id, tokens, semantic_key, plan)
                if partial is not None:
                    return self._record(partial)
                if not self.allow_approximate_semantic_reuse and self.semantic_ablation_mode == 'safe':
                    # HF/real backends intentionally block approximate semantic
                    # KV reuse for correctness. Treat the request as a safe
                    # bypass after recording opportunity metrics; do not pay an
                    # unnecessary exact-prefix store cost for a non-exact match.
                    self.bank.record_miss()
                    out = self._prefill_full(tokens)
                    self.controller.update_feedback(hit=False, wasted_ratio=0.0)
                    return self._record(RequestResult(request_id, out.latency_ms, 0, len(tokens), False, False, gpu_utilization_pct=out.gpu_utilization_pct))
                if self.semantic_ablation_mode in {'scaffold_only', 'early_layer', 'logit_guard'}:
                    self.bank.record_miss()
                    out = self._prefill_full(tokens)
                    store_latency_ms = self._store_semantic_scaffold_prefix(tokens, metadata=metadata, tier='cpu')
                    self.controller.update_feedback(hit=False, wasted_ratio=0.0)
                    return self._record(RequestResult(request_id, out.latency_ms + store_latency_ms, 0, len(tokens), False, False, gpu_utilization_pct=out.gpu_utilization_pct))
            if match is None:
                self.bank.record_miss()
                out = self._prefill_full(tokens)
                store_latency_ms = self._store_reactive_prefix(tokens, tier='cpu', metadata=metadata) if self.cache_enabled else 0.0
                self.controller.update_feedback(hit=False, wasted_ratio=0.0)
                return self._record(RequestResult(request_id, out.latency_ms + store_latency_ms, 0, len(tokens), False, False, gpu_utilization_pct=out.gpu_utilization_pct))
            key, entry, match_len = match
            if plan.strategy != 'exact' or not self._should_attempt_cache_use(tokens, entry, match_len, metadata=metadata):
                self.bank.record_miss()
                out = self._prefill_full(tokens)
                store_latency_ms = self._store_reactive_prefix(tokens, tier='cpu', metadata=metadata) if self.cache_enabled else 0.0
                self.controller.update_feedback(hit=False, wasted_ratio=0.0)
                return self._record(RequestResult(request_id, out.latency_ms + store_latency_ms, 0, len(tokens), False, False, gpu_utilization_pct=out.gpu_utilization_pct))
            admitted = self.bank.admit_match(key)
            if admitted is None:
                self.bank.record_miss(); out = self._prefill_full(tokens); return self._record(RequestResult(request_id, out.latency_ms, 0, len(tokens), False, False, gpu_utilization_pct=out.gpu_utilization_pct))
            try:
                setattr(admitted, 'layer_reuse_ratio', plan.layer_reuse_ratio)
            except Exception:
                pass
            out, recomputed, hit, actual = self._prefill_with_cache_fallback(tokens, admitted, match_len)
            self.controller.update_feedback(hit=hit, wasted_ratio=0.0)
            return self._record(RequestResult(request_id, out.latency_ms, actual if hit else 0, recomputed, hit, admitted.was_speculative if hit else False, cache_tier=admitted.tier if hit else None, gpu_utilization_pct=out.gpu_utilization_pct))
        finally:
            self.serving_event.clear()

def maybe_shutdown(engine: BaseEngine) -> None:
    if hasattr(engine, 'shutdown'):
        engine.shutdown()  # type: ignore[attr-defined]
    else:
        engine.finalize()

    if hasattr(engine, 'bank') and hasattr(engine.bank, 'finalize_run'):
        engine.bank.finalize_run()


def summarize_engine(engine: BaseEngine) -> Dict:
    speculative_log = getattr(engine, 'speculative_log', [])
    summary = summarize_run(
        latencies=engine.latencies,
        gpu_utils=engine.gpu_utils,
        bank_metrics=engine.bank.snapshot_metrics(),
        speculative_precomputes=len(speculative_log),
        speculative_cost_ms=sum(x['latency_ms'] for x in speculative_log),
        total_wall_time_s=max(engine.end_time - engine.start_time, 1e-9),
        extra_metrics=engine.engine_metrics,
    ).to_dict()
    # Preserve experimental controller metrics without forcing every metric into
    # the stable RunSummary dataclass. This keeps older reports compatible while
    # making ShadowKV++ diagnostics visible in JSON outputs.
    for key, value in engine.engine_metrics.items():
        summary.setdefault(key, value)
    return summary


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
