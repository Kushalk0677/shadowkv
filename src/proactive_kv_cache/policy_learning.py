from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ENGINE_KEYS = {
    'no_cache',
    'native_prefix_cache',
    'reactive_prefix_cache',
    'greedy_prefix_cache',
    'strict_reactive_prefix_cache',
    'frequency_speculative',
    'shadow_kv',
    'shadow_kv_plus',
    'vllm_apc',
    'vllm_apc_shadowkv_plus',
    'sglang_radix_attention',
    'sglang_radix_attention_shadowkv_plus',
    'lmcache',
    'lmcache_shadowkv_plus',
}


@dataclass(frozen=True)
class RunFeatureRow:
    source: str
    engine: str
    workload: str
    dataset: str
    prompt_mode: str
    n_requests: int
    mean_latency_ms: float
    speedup_vs_no_cache: float
    cache_hit_rate: float
    wasted_compute_ratio: float
    reused_prefix_tokens_total: float
    recompute_tokens_total: float
    policy_net_utility_ms: float = 0.0
    semantic_partial_hits: float = 0.0
    semantic_opportunity_plans_total: float = 0.0
    semantic_opportunity_estimated_savings_ms: float = 0.0
    semantic_blocked_by_backend_total: float = 0.0
    layer_reuse_events: float = 0.0

    @property
    def reuse_density(self) -> float:
        denom = self.reused_prefix_tokens_total + self.recompute_tokens_total
        return 0.0 if denom <= 0 else self.reused_prefix_tokens_total / denom


def _iter_json_payloads(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if path.is_dir():
        for child in sorted(path.rglob('*.json')):
            try:
                yield str(child), json.loads(child.read_text())
            except Exception:
                continue
        for child in sorted(path.rglob('*.zip')):
            yield from _iter_json_payloads(child)
        return
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith('.json'):
                    continue
                try:
                    yield f'{path}!{name}', json.loads(zf.read(name).decode('utf-8'))
                except Exception:
                    continue
        return
    if path.suffix.lower() == '.json':
        yield str(path), json.loads(path.read_text())


def load_feature_rows(paths: Iterable[str | Path]) -> List[RunFeatureRow]:
    rows: List[RunFeatureRow] = []
    for raw_path in paths:
        for source, payload in _iter_json_payloads(Path(raw_path)):
            config = payload.get('config', {}) if isinstance(payload, dict) else {}
            no_cache = payload.get('no_cache', {}) if isinstance(payload, dict) else {}
            base_mean = float(no_cache.get('mean_latency_ms') or no_cache.get('latency_mean_ms') or 0.0)
            for engine, metrics in payload.items():
                if engine not in ENGINE_KEYS or not isinstance(metrics, dict):
                    continue
                mean = float(metrics.get('mean_latency_ms') or metrics.get('latency_mean_ms') or 0.0)
                if mean <= 0:
                    continue
                speedup = float(metrics.get('speedup_vs_no_cache_mean') or (base_mean / mean if base_mean > 0 else 1.0))
                rows.append(
                    RunFeatureRow(
                        source=source,
                        engine=engine,
                        workload=str(config.get('workload', 'unknown')),
                        dataset=str(config.get('dataset') or config.get('variant') or 'unknown'),
                        prompt_mode=str(config.get('resolved_prompt_mode') or config.get('prompt_mode') or 'unknown'),
                        n_requests=int(config.get('n_requests') or 0),
                        mean_latency_ms=mean,
                        speedup_vs_no_cache=speedup,
                        cache_hit_rate=float(metrics.get('cache_hit_rate') or 0.0),
                        wasted_compute_ratio=float(metrics.get('wasted_compute_ratio') or 0.0),
                        reused_prefix_tokens_total=float(metrics.get('reused_prefix_tokens_total') or 0.0),
                        recompute_tokens_total=float(metrics.get('recompute_tokens_total') or 0.0),
                        policy_net_utility_ms=float(metrics.get('policy_net_utility_ms') or 0.0),
                        semantic_partial_hits=float(metrics.get('semantic_partial_hits') or 0.0),
                        semantic_opportunity_plans_total=float(metrics.get('semantic_opportunity_plans_total') or 0.0),
                        semantic_opportunity_estimated_savings_ms=float(metrics.get('semantic_opportunity_estimated_savings_ms') or 0.0),
                        semantic_blocked_by_backend_total=float(metrics.get('semantic_blocked_by_backend_total') or 0.0),
                        layer_reuse_events=float(metrics.get('layer_reuse_events') or 0.0),
                    )
                )
    return rows


def learn_shadowkv_plus_thresholds(rows: Iterable[RunFeatureRow]) -> Dict[str, float]:
    """Learn a small deployment policy from completed benchmark logs.

    This deliberately stays dependency-free. It searches thresholds that predict
    whether an adaptive KV engine should be enabled for a workload family.

    The target label is speedup_vs_no_cache > 1.02 and wasted_compute_ratio < 0.35.
    Returned values can be used as controller defaults or as reporting evidence.
    """

    # Learn deployment gates from ShadowKV++ evidence only.  ShadowKV and
    # ShadowKV++ have different mechanisms and waste profiles; mixing the old
    # engine into the training target makes the learned gate look better than it
    # really is.  If there are not enough ShadowKV++ rows yet, return a
    # conservative default and make the insufficiency explicit.
    data = [r for r in rows if r.engine == 'shadow_kv_plus']
    if len(data) < 8:
        return {
            'min_reuse_density': 0.03,
            'max_waste_ratio': 0.35,
            'min_cache_hit_rate': 0.05,
            'estimated_accuracy': 0.0,
            'n_training_rows': int(len(data)),
            'status': 'insufficient_shadowkv_plus_rows',
        }

    candidates_density = sorted({0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, *[round(r.reuse_density, 3) for r in data]})
    candidates_waste = sorted({0.10, 0.20, 0.30, 0.35, 0.45, 0.60, *[round(r.wasted_compute_ratio, 3) for r in data]})
    candidates_hit = sorted({0.0, 0.02, 0.05, 0.08, 0.12, *[round(r.cache_hit_rate, 3) for r in data]})

    best = (-1.0, 0.03, 0.35, 0.05)
    labels = [(r.speedup_vs_no_cache > 1.02 and r.wasted_compute_ratio < 0.35) for r in data]

    for d in candidates_density:
        for w in candidates_waste:
            for h in candidates_hit:
                correct = 0
                for r, label in zip(data, labels):
                    pred = (r.reuse_density >= d) and (r.wasted_compute_ratio <= w) and (r.cache_hit_rate >= h)
                    correct += int(pred == label)
                acc = correct / max(len(data), 1)
                # Tie-break toward conservative, lower-waste policies.
                score = acc - 0.002 * w + 0.001 * d + 0.001 * h
                if score > best[0]:
                    best = (score, d, w, h)

    score, d, w, h = best
    return {
        'min_reuse_density': float(d),
        'max_waste_ratio': float(w),
        'min_cache_hit_rate': float(h),
        'estimated_accuracy': float(max(min(score, 1.0), 0.0)),
        'n_training_rows': int(len(data)),
    }


def rows_to_csv(rows: Iterable[RunFeatureRow]) -> str:
    header = [
        'source', 'engine', 'workload', 'dataset', 'prompt_mode', 'n_requests',
        'mean_latency_ms', 'speedup_vs_no_cache', 'cache_hit_rate',
        'wasted_compute_ratio', 'reuse_density', 'reused_prefix_tokens_total',
        'recompute_tokens_total', 'policy_net_utility_ms',
        'semantic_partial_hits', 'semantic_opportunity_plans_total',
        'semantic_opportunity_estimated_savings_ms', 'semantic_blocked_by_backend_total',
        'layer_reuse_events',
    ]
    lines = [','.join(header)]
    for r in rows:
        vals = [
            r.source, r.engine, r.workload, r.dataset, r.prompt_mode, str(r.n_requests),
            f'{r.mean_latency_ms:.6f}', f'{r.speedup_vs_no_cache:.6f}',
            f'{r.cache_hit_rate:.6f}', f'{r.wasted_compute_ratio:.6f}',
            f'{r.reuse_density:.6f}', f'{r.reused_prefix_tokens_total:.1f}',
            f'{r.recompute_tokens_total:.1f}', f'{r.policy_net_utility_ms:.6f}',
            f'{r.semantic_partial_hits:.1f}', f'{r.semantic_opportunity_plans_total:.1f}',
            f'{r.semantic_opportunity_estimated_savings_ms:.6f}',
            f'{r.semantic_blocked_by_backend_total:.1f}', f'{r.layer_reuse_events:.1f}',
        ]
        escaped = []
        for v in vals:
            s = str(v).replace('"', '""')
            escaped.append(f'"{s}"' if ',' in s or '\n' in s else s)
        lines.append(','.join(escaped))
    return '\n'.join(lines) + '\n'
