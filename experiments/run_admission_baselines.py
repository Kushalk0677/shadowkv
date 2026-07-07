#!/usr/bin/env python3
"""Admission-control baselines sweep.

Compares MeritKV against simpler gating strategies on the same workload.
Run on FakeBackend (CPU) — ~30 min for all baselines.

Usage:
    python experiments/run_admission_baselines.py
    python experiments/run_admission_baselines.py --quick   # 3 datasets, 20 requests
"""
import argparse, json, os, sys, time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

from proactive_kv_cache.config_loader import CONFIG
from proactive_kv_cache.engines import (
    ShadowKVPlusEngine, ShadowKVEngine, NoCacheEngine,
    ReactivePrefixCacheEngine, StrictReactivePrefixCacheEngine,
    GreedyPrefixCacheEngine, FrequencySpeculativeEngine,
    summarize_engine, maybe_shutdown,
)
from proactive_kv_cache.models import FakeBackend
from proactive_kv_cache.utility import UtilityModel, AdmissionEvent

# ---- Baseline definitions ----
BASELINES = []

def add(name, engine_cls, tuning_overrides=None, engine_kwargs=None):
    BASELINES.append({
        'name': name,
        'cls': engine_cls,
        'tuning': tuning_overrides or {},
        'kwargs': engine_kwargs or {},
    })

# MeritKV (reference)
add('MeritKV', ShadowKVPlusEngine, {},
    {'allow_approximate_semantic_reuse': False})

# ShadowKV (prior system, no waste term)
add('ShadowKV (no waste)', ShadowKVEngine, {})

# Prefix length gates — reuse only if match >= threshold
for threshold in [16, 32, 48, 64]:
    add(f'Gate: len>={threshold}',
        ReactivePrefixCacheEngine,
        {'min_reuse_prefix_tokens': threshold,
         'min_store_prefix_tokens': max(threshold - 8, 3)})

# Cost-only gate (B - C >= 0, W=0)
# We create this by overriding UtilityModel via config
add('Gate: cost-only', ShadowKVPlusEngine,
    {},
    {'allow_approximate_semantic_reuse': False})

# Frequency threshold (reuse after n observations)
add('Gate: freq>=3', FrequencySpeculativeEngine, {},
    {'speculative_k': 0, 'idle_threshold_ms': 999999})

# Strict reactive (tighter thresholds, no speculation)
add('Gate: strict', StrictReactivePrefixCacheEngine, {})

# Greedy (reuse everything)
add('Gate: greedy', GreedyPrefixCacheEngine, {})

# No cache (baseline floor)
add('No cache', NoCacheEngine, {})

# ---- Workload ----
WORKLOADS = {
    'templated': {
        'prompt_mode': 'templated',
        'n_requests': 80,
    },
    'semantic': {
        'prompt_mode': 'semantic',
        'n_requests': 80,
    },
}


def make_requests(n, prompt_mode, seed=42):
    """Generate requests with realistic prefix sharing."""
    import random
    rng = random.Random(seed)
    
    class Req:
        def __init__(self, rid, prompt, metadata):
            self.request_id = rid
            self.prompt = prompt
            self.metadata = metadata
            self.arrival_time = 0.0
    
    shared = (
        "You are a helpful AI assistant. Please answer the following question "
        "with a detailed, well-structured response. Provide examples where "
        "appropriate and keep your answer clear and concise.\n\n"
    )
    
    topics = [
        "machine learning", "quantum computing", "ancient Rome", "climate change",
        "the human brain", "cryptocurrency", "photosynthesis", "World War II",
        "renewable energy", "DNA", "blockchain", "the Solar System",
        "the French Revolution", "artificial intelligence", "the Great Depression",
    ]
    
    requests = []
    for i in range(n):
        topic = topics[i % len(topics)]
        suffix = f"Tell me about {topic}. Please include at least three key points."
        prompt = shared + suffix
        
        if prompt_mode == 'semantic':
            # Paraphrase the suffix for semantic mode
            variants = [
                f"Explain {topic} in detail.",
                f"What do you know about {topic}?",
                f"Describe {topic} with examples.",
                f"I want to learn about {topic}.",
                f"Give me information on {topic}.",
            ]
            suffix = variants[i % len(variants)]
            prompt = shared + suffix
        
        metadata = {
            'prompt_mode': prompt_mode,
            'shared_prefix_hint_tokens': len(shared.split()),
            'shared_prefix_text': shared,
        }
        requests.append(Req(i, prompt, metadata))
    
    return requests


def tokenize_batch(backend, requests):
    for req in requests:
        req.tokens = tuple(backend.tokenize(req.prompt))


def run_one(engine, requests):
    for req in requests:
        engine.serve_tokens(req.request_id, req.tokens, metadata=req.metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='20 requests, 1 dataset')
    args = parser.parse_args()
    
    backend = FakeBackend(device='cpu')
    
    all_results = []
    
    for wl_name, wl_cfg in WORKLOADS.items():
        if args.quick and wl_name != 'templated':
            continue
        
        n = 20 if args.quick else wl_cfg['n_requests']
        requests = make_requests(n, wl_cfg['prompt_mode'])
        tokenize_batch(backend, requests)
        
        print(f"\n{'='*60}")
        print(f"  Workload: {wl_name} ({n} requests)")
        print(f"{'='*60}")
        print(f"  {'Engine':>25s}  {'Speedup':>7s}  {'Waste':>6s}  {'HitRate':>7s}")
        print(f"  {'-'*25}  {'-'*7}  {'-'*6}  {'-'*7}")
        
        for bl in BASELINES:
            name = bl['name']
            
            # Apply config overrides for cost-only gate
            if name == 'Gate: cost-only':
                # Zero out the waste weight in health model
                CONFIG.update({
                    'policy.health.waste_ratio_weight': 0.0,
                    'policy.health.hit_rate_weight': 0.0,
                    'policy.health.base': 1.0,
                    'policy.utility.semantic_prompt_waste_floor': 0.0,
                    'policy.utility.semantic_default_waste_floor': 0.0,
                })
            else:
                # Reset to defaults
                CONFIG.update({
                    'policy.health.waste_ratio_weight': 0.45,
                    'policy.health.hit_rate_weight': 0.35,
                    'policy.health.base': 0.65,
                    'policy.utility.semantic_prompt_waste_floor': 0.03,
                    'policy.utility.semantic_default_waste_floor': 0.05,
                })
            
            CONFIG._cached_get_results.clear()
            
            # Run no-cache baseline
            no_cache = NoCacheEngine(backend=backend, max_memory_mb=256)
            run_one(no_cache, requests)
            no_cache.finalize()
            baseline = summarize_engine(no_cache)
            
            # Run baseline engine
            kwargs = dict(bl['kwargs'])
            if bl['cls'] == ShadowKVPlusEngine:
                kwargs['allow_approximate_semantic_reuse'] = False
            
            engine = bl['cls'](
                backend=backend,
                max_memory_mb=256,
                **kwargs
            )
            
            # Apply tuning overrides
            for k, v in bl['tuning'].items():
                setattr(engine.tuning, k, v)
            
            run_one(engine, requests)
            maybe_shutdown(engine)
            summary = summarize_engine(engine)
            
            speedup = baseline.get('mean_latency_ms', 1.0) / max(summary.get('mean_latency_ms', 0.001), 0.001)
            waste = summary.get('waste_ratio', 0.0)
            hit_rate = summary.get('hit_rate', 0.0)
            
            print(f"  {name:>25s}  {speedup:>7.3f}x  {waste:>6.3f}  {hit_rate:>7.3f}")
            
            all_results.append({
                'workload': wl_name,
                'engine': name,
                'speedup': round(speedup, 4),
                'waste': round(waste, 4),
                'hit_rate': round(hit_rate, 4),
                'mean_latency_ms': round(summary.get('mean_latency_ms', 0.0), 2),
            })
    
    # Save
    out_dir = ROOT / 'results' / 'admission_baselines'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'baseline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY — MeritKV vs Simple Gates")
    print(f"{'='*70}")
    print(f"  {'Engine':>25s}  {'Templated':>20s}  {'Semantic':>20s}")
    print(f"  {'':>25s}  {'speedup  waste  hit':>20s}  {'speedup  waste  hit':>20s}")
    print(f"  {'-'*25}  {'-'*20}  {'-'*20}")
    
    for bl in BASELINES:
        name = bl['name']
        t_row = [r for r in all_results if r['engine'] == name and r['workload'] == 'templated']
        s_row = [r for r in all_results if r['engine'] == name and r['workload'] == 'semantic']
        
        t_str = f"{t_row[0]['speedup']:.3f}x {t_row[0]['waste']:.3f} {t_row[0]['hit_rate']:.3f}" if t_row else "---"
        s_str = f"{s_row[0]['speedup']:.3f}x {s_row[0]['waste']:.3f} {s_row[0]['hit_rate']:.3f}" if s_row else "---"
        
        print(f"  {name:>25s}  {t_str:>20s}  {s_str:>20s}")
    
    print(f"\n  Saved: {out_dir / 'baseline_results.json'}")


if __name__ == '__main__':
    main()
