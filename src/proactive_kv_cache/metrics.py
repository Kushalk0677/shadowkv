from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


@dataclass
class RunSummary:
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    p999_latency_ms: float
    throughput_rps: float
    service_throughput_rps: float
    hit_rate: float
    speculative_hits: int
    speculative_hit_rate: float
    speculative_precomputes: int
    speculative_cost_ms: float
    wasted_precomputes: int
    wasted_compute_ms: float
    wasted_compute_ratio: float
    utility_proxy_ms: float
    gpu_utilization_pct_mean: float | None
    promotions: int
    demotions: int
    reuse_attempts: int = 0
    reuse_successes: int = 0
    reuse_failures: int = 0
    reuse_backend_fallbacks: int = 0
    store_attempts: int = 0
    store_successes: int = 0
    store_skips: int = 0
    bypassed_matches: int = 0
    requests_seen: int = 0
    reused_prefix_tokens_total: int = 0
    recompute_tokens_total: int = 0
    estimated_tokens_saved_total: int = 0
    saved_latency_estimate_ms: float = 0.0
    store_latency_total_ms: float = 0.0
    full_prefill_latency_total_ms: float = 0.0
    avg_reused_prefix_tokens: float = 0.0
    reuse_success_rate: float = 0.0
    cache_active_final: bool = True
    auto_disabled_reason: str | None = None
    speculative_overlap_ms: float = 0.0
    speculative_overlap_events: int = 0
    bootstrap_store_deferrals: int = 0
    speculative_useful_savings_ms: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def summarize_run(
    latencies: List[float],
    gpu_utils: List[float | None],
    bank_metrics: Dict,
    speculative_precomputes: int,
    speculative_cost_ms: float,
    total_wall_time_s: float,
    extra_metrics: Dict | None = None,
) -> RunSummary:
    lat = np.asarray(latencies, dtype=float)

    mean_latency = float(lat.mean()) if lat.size else 0.0
    p50 = float(np.percentile(lat, 50)) if lat.size else 0.0
    p95 = float(np.percentile(lat, 95)) if lat.size else 0.0
    p99 = float(np.percentile(lat, 99)) if lat.size else 0.0
    p999 = float(np.percentile(lat, 99.9)) if lat.size else 0.0

    speculative_hits = int(bank_metrics.get('speculative_hits', 0))
    speculative_hit_rate = float(bank_metrics.get('speculative_hit_rate', 0.0))
    hit_rate = float(bank_metrics.get('hit_rate', 0.0))

    wasted_precomputes = int(bank_metrics.get('wasted_precomputes', 0))
    if 'wasted_compute_ms' in bank_metrics:
        wasted_compute_ms = float(bank_metrics['wasted_compute_ms'])
    else:
        wasted_ratio_fallback = wasted_precomputes / max(speculative_precomputes, 1)
        wasted_compute_ms = float(speculative_cost_ms * wasted_ratio_fallback)
    wasted_compute_ratio = float(min(wasted_compute_ms / max(speculative_cost_ms, 1e-9), 1.0))

    if 'useful_speculative_savings_ms' in bank_metrics:
        utility_proxy = float(bank_metrics['useful_speculative_savings_ms'] - wasted_compute_ms)
    else:
        utility_proxy = float(speculative_hits * mean_latency - wasted_compute_ms)

    gpu_values = [x for x in gpu_utils if x is not None]
    gpu_mean = float(np.mean(gpu_values)) if gpu_values else None

    throughput = float(len(latencies) / max(total_wall_time_s, 1e-9))
    service_time_s = float(lat.sum() / 1000.0)
    service_throughput = float(len(latencies) / max(service_time_s, 1e-9))

    merged = dict(extra_metrics or {})
    for key in (
        'reuse_attempts', 'reuse_successes', 'reuse_failures', 'reuse_backend_fallbacks', 'store_attempts', 'store_successes', 'store_skips', 'bypassed_matches',
        'requests_seen', 'reused_prefix_tokens_total', 'recompute_tokens_total', 'estimated_tokens_saved_total'
    ):
        merged[key] = int(merged.get(key, bank_metrics.get(key, 0)))
    for key in ('saved_latency_estimate_ms', 'store_latency_total_ms', 'full_prefill_latency_total_ms'):
        merged[key] = float(merged.get(key, bank_metrics.get(key, 0.0)))
    merged['cache_active_final'] = bool(merged.get('cache_active_final', True))
    merged['auto_disabled_reason'] = merged.get('auto_disabled_reason')
    merged['avg_reused_prefix_tokens'] = float(merged['reused_prefix_tokens_total'] / max(merged['reuse_successes'], 1))
    merged['reuse_success_rate'] = float(merged['reuse_successes'] / max(merged['reuse_successes'] + merged['reuse_failures'], 1))
    merged['speculative_overlap_ms'] = float(merged.get('speculative_overlap_ms', 0.0))
    merged['speculative_overlap_events'] = int(merged.get('speculative_overlap_events', 0))
    merged['bootstrap_store_deferrals'] = int(merged.get('bootstrap_store_deferrals', 0))
    merged['speculative_useful_savings_ms'] = float(merged.get('speculative_useful_savings_ms', bank_metrics.get('useful_speculative_savings_ms', 0.0)))

    return RunSummary(
        mean_latency_ms=mean_latency,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        p999_latency_ms=p999,
        throughput_rps=throughput,
        service_throughput_rps=service_throughput,
        hit_rate=hit_rate,
        speculative_hits=speculative_hits,
        speculative_hit_rate=speculative_hit_rate,
        speculative_precomputes=int(speculative_precomputes),
        speculative_cost_ms=float(speculative_cost_ms),
        wasted_precomputes=wasted_precomputes,
        wasted_compute_ms=wasted_compute_ms,
        wasted_compute_ratio=wasted_compute_ratio,
        utility_proxy_ms=utility_proxy,
        gpu_utilization_pct_mean=gpu_mean,
        promotions=int(bank_metrics.get('promotions', 0)),
        demotions=int(bank_metrics.get('demotions', 0)),
        reuse_attempts=merged['reuse_attempts'],
        reuse_successes=merged['reuse_successes'],
        reuse_failures=merged['reuse_failures'],
        reuse_backend_fallbacks=merged['reuse_backend_fallbacks'],
        store_attempts=merged['store_attempts'],
        store_successes=merged['store_successes'],
        store_skips=merged['store_skips'],
        bypassed_matches=merged['bypassed_matches'],
        requests_seen=merged['requests_seen'],
        reused_prefix_tokens_total=merged['reused_prefix_tokens_total'],
        recompute_tokens_total=merged['recompute_tokens_total'],
        estimated_tokens_saved_total=merged['estimated_tokens_saved_total'],
        saved_latency_estimate_ms=merged['saved_latency_estimate_ms'],
        store_latency_total_ms=merged['store_latency_total_ms'],
        full_prefill_latency_total_ms=merged['full_prefill_latency_total_ms'],
        avg_reused_prefix_tokens=merged['avg_reused_prefix_tokens'],
        reuse_success_rate=merged['reuse_success_rate'],
        cache_active_final=merged['cache_active_final'],
        auto_disabled_reason=merged['auto_disabled_reason'],
        speculative_overlap_ms=merged['speculative_overlap_ms'],
        speculative_overlap_events=merged['speculative_overlap_events'],
        bootstrap_store_deferrals=merged['bootstrap_store_deferrals'],
        speculative_useful_savings_ms=merged['speculative_useful_savings_ms'],
    )
