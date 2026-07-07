#!/usr/bin/env python3
"""Mixed-traffic workload generator for admission-control evaluation.

Generates interleaved request streams combining raw, templated, RAG, and
semantic modes, then runs them through all engines on FakeBackend.

Usage:
    python experiments/run_mixed_traffic.py --workload mixed_serving --n_requests 100
    python experiments/run_mixed_traffic.py --workload all
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

from proactive_kv_cache.engines import (
    ShadowKVEngine, ShadowKVPlusEngine, NoCacheEngine, ReactivePrefixCacheEngine,
    StrictReactivePrefixCacheEngine, GreedyPrefixCacheEngine,
    FrequencySpeculativeEngine, summarize_engine, maybe_shutdown,
)
from proactive_kv_cache.models import FakeBackend

# ---- Workload Mix Definitions ----
# Each workload defines (ratio_raw, ratio_templated, ratio_rag, ratio_semantic)
# summing to 1.0. The __suffix__ controls how the workload variant is generated.

MIXED_WORKLOADS = {
    'clean_reusable': {
        'ratios': (0.05, 0.85, 0.10, 0.0),
        'desc': '90% templated/RAG with long shared prefixes',
        'suffix_kwargs': {'long_prefix_bias': 1.5, 'mean_inter_arrival_ms': 100},
    },
    'mostly_raw': {
        'ratios': (0.90, 0.05, 0.05, 0.0),
        'desc': '90% diverse raw prompts, minimal reuse',
        'suffix_kwargs': {'alpha': 0.3, 'mean_inter_arrival_ms': 80},
    },
    'mixed_serving': {
        'ratios': (0.35, 0.30, 0.25, 0.10),
        'desc': 'Realistic chat + RAG + semantic mix',
        'suffix_kwargs': {'alpha': 1.0, 'mean_inter_arrival_ms': 120, 'burst_probability': 0.15},
    },
    'bursty_reuse': {
        'ratios': (0.20, 0.60, 0.20, 0.0),
        'desc': 'Repeated prefix in bursts, then disappears',
        'suffix_kwargs': {'alpha': 1.5, 'mean_inter_arrival_ms': 150, 'burst_probability': 0.40},
    },
    'adversarial_short': {
        'ratios': (0.10, 0.70, 0.20, 0.0),
        'desc': 'Many apparent matches below breakeven (k<16)',
        'suffix_kwargs': {'alpha': 2.0, 'mean_inter_arrival_ms': 100,
                          'short_prefix_bias': 1.0},
    },
    'speculation_trap': {
        'ratios': (0.10, 0.70, 0.20, 0.0),
        'desc': 'High early frequency, then no future reuse',
        'suffix_kwargs': {'alpha': 0.5, 'mean_inter_arrival_ms': 80,
                          'trap_prefix_decay': 0.85},
    },
}

# ---- Engine list (same as admission baselines) ----
ENGINES = [
    ('No cache', NoCacheEngine, {}, {}),
    ('MeritKV', ShadowKVPlusEngine, {}, {'allow_approximate_semantic_reuse': False}),
    ('ShadowKV (no waste)', ShadowKVEngine, {}, {}),
    ('Gate: len>=16', ReactivePrefixCacheEngine, {'min_reuse_prefix_tokens': 16, 'min_store_prefix_tokens': 8}, {}),
    ('Gate: len>=32', ReactivePrefixCacheEngine, {'min_reuse_prefix_tokens': 32, 'min_store_prefix_tokens': 24}, {}),
    ('Gate: len>=48', ReactivePrefixCacheEngine, {'min_reuse_prefix_tokens': 48, 'min_store_prefix_tokens': 40}, {}),
    ('Gate: strict', StrictReactivePrefixCacheEngine, {}, {}),
    ('Gate: greedy', GreedyPrefixCacheEngine, {}, {}),
]


def make_mixed_workload(workload_id, n_requests, seed=42):
    """Generate an interleaved mixed-traffic workload."""
    import random
    rng = random.Random(seed)
    wl = MIXED_WORKLOADS[workload_id]
    ratios = wl['ratios']  # raw, templated, rag, semantic
    
    class Req:
        def __init__(self, rid, prompt, metadata):
            self.request_id = rid
            self.prompt = prompt
            self.metadata = metadata
            self.arrival_time = 0.0
    
    # Shared templates for each mode
    raw_suffixes = [
        "What is the capital of Australia?",
        "How does photosynthesis work?",
        "Why is the sky blue?",
        "Write a Python function to reverse a linked list.",
        "Explain quantum entanglement.",
    ]
    templated_shared = (
        "You are a helpful assistant. Answer concisely: "
    )
    rag_shared = (
        "System: You are a RAG assistant. Use the provided context to answer.\n"
        "Context: The deployment requires canary, health checks, rollback.\n"
        "Question: "
    )
    semantic_shared = (
        "Classify the following item by dominant topic. Return one label.\nItem:\n"
    )
    semantic_items = [
        "The central bank kept rates unchanged.",
        "A new battery design improved EV range.",
        "The football club signed a young striker.",
        "Researchers released a compact LLM for summarization.",
    ]
    
    topics = [
        "machine learning", "quantum computing", "ancient Rome", "climate change",
        "the human brain", "cryptocurrency", "photosynthesis", "World War II",
        "renewable energy", "DNA", "blockchain", "the Solar System",
    ]
    
    requests = []
    for i in range(n_requests):
        # Sample mode from ratios
        mode_rnd = rng.random()
        cum = 0
        mode_idx = 0
        for j, r in enumerate(ratios):
            cum += r
            if mode_rnd < cum:
                mode_idx = j
                break
        
        if mode_idx == 0:  # raw
            t = raw_suffixes[i % len(raw_suffixes)]
            prompt = t
            metadata = {'prompt_mode': 'raw'}
        elif mode_idx == 1:  # templated
            t = topics[i % len(topics)]
            prompt = f"{templated_shared}Tell me about {t}."
            metadata = {
                'prompt_mode': 'templated',
                'shared_prefix_hint_tokens': len(templated_shared.split()),
                'shared_prefix_text': templated_shared,
            }
        elif mode_idx == 2:  # rag
            q = raw_suffixes[i % len(raw_suffixes)]
            prompt = f"{rag_shared}{q}"
            metadata = {
                'prompt_mode': 'rag',
                'shared_prefix_hint_tokens': len(rag_shared.split()),
                'shared_prefix_text': rag_shared,
            }
        else:  # semantic
            item = semantic_items[i % len(semantic_items)]
            prompt = f"{semantic_shared}{item}\nCategory:"
            metadata = {
                'prompt_mode': 'semantic',
                'shared_prefix_hint_tokens': len(semantic_shared.split()),
                'shared_prefix_text': semantic_shared,
            }
        
        # Speculation trap: after mid-point, mask sharing
        if 'trap_prefix_decay' in wl.get('suffix_kwargs', {}):
            if i > n_requests // 2:
                metadata['prompt_mode'] = 'raw'
                metadata.pop('shared_prefix_hint_tokens', None)
                metadata.pop('shared_prefix_text', None)
        
        # Adversarial short: use very short shared prefix
        if 'short_prefix_bias' in wl.get('suffix_kwargs', {}):
            if 'shared_prefix_text' in metadata:
                metadata['shared_prefix_hint_tokens'] = min(
                    metadata.get('shared_prefix_hint_tokens', 999), 8)
        
        requests.append(Req(i, prompt, metadata))
    
    return requests


def tokenize_batch(backend, requests):
    for req in requests:
        req.tokens = tuple(backend.tokenize(req.prompt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', choices=list(MIXED_WORKLOADS.keys()) + ['all'], default='all')
    parser.add_argument('--n_requests', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    workloads = list(MIXED_WORKLOADS.keys()) if args.workload == 'all' else [args.workload]
    backend = FakeBackend(device='cpu')
    all_results = []
    
    for wl_id in workloads:
        wl = MIXED_WORKLOADS[wl_id]
        n = args.n_requests
        requests = make_mixed_workload(wl_id, n, args.seed)
        tokenize_batch(backend, requests)
        
        print(f"\n{'='*60}")
        print(f"  {wl_id}: {wl['desc']} ({n} requests)")
        print(f"{'='*60}")
        print(f"  {'Engine':>25s}  {'Speedup':>7s}  {'Waste':>6s}  {'HitRate':>7s}")
        print(f"  {'-'*25}  {'-'*7}  {'-'*6}  {'-'*7}")
        
        for name, cls, tuning, kwargs in ENGINES:
            # No-cache baseline
            no_cache = NoCacheEngine(backend=backend, max_memory_mb=256)
            for req in requests:
                no_cache.serve_tokens(req.request_id, req.tokens, metadata=req.metadata)
            no_cache.finalize()
            baseline = summarize_engine(no_cache)
            
            # Engine
            eng = cls(backend=backend, max_memory_mb=256, **kwargs)
            for k, v in tuning.items():
                setattr(eng.tuning, k, v)
            for req in requests:
                eng.serve_tokens(req.request_id, req.tokens, metadata=req.metadata)
            maybe_shutdown(eng)
            summary = summarize_engine(eng)
            
            speedup = baseline.get('mean_latency_ms', 1.0) / max(summary.get('mean_latency_ms', 0.001), 0.001)
            waste = summary.get('waste_ratio', 0.0)
            hit_rate = summary.get('hit_rate', 0.0)
            
            print(f"  {name:>25s}  {speedup:>7.3f}x  {waste:>6.3f}  {hit_rate:>7.3f}")
            
            all_results.append({
                'workload': wl_id,
                'engine': name,
                'speedup': round(speedup, 4),
                'waste': round(waste, 4),
                'hit_rate': round(hit_rate, 4),
            })
    
    # Save
    out_dir = ROOT / 'results' / 'mixed_traffic'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'mixed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("  MIXED TRAFFIC SUMMARY")
    print(f"{'='*70}")
    wl_col = 18
    print(f"  {'Workload':>{wl_col}s}  {'MeritKV':>16s}  {'No waste':>14s}  {'Len>=16':>14s}  {'Strict':>14s}")
    print(f"  {'':>{wl_col}s}  {'spd  waste hit':>16s}  {'spd  waste hit':>14s}  {'spd  waste hit':>14s}  {'spd  waste hit':>14s}")
    print(f"  {'-'*wl_col}  {'-'*16}  {'-'*14}  {'-'*14}  {'-'*14}")
    
    for wl_id in workloads:
        row_mk = [r for r in all_results if r['workload']==wl_id and r['engine']=='MeritKV']
        row_nw = [r for r in all_results if r['workload']==wl_id and r['engine']=='ShadowKV (no waste)']
        row_l16 = [r for r in all_results if r['workload']==wl_id and r['engine']=='Gate: len>=16']
        row_st = [r for r in all_results if r['workload']==wl_id and r['engine']=='Gate: strict']
        mk = f"{row_mk[0]['speedup']:.3f} {row_mk[0]['waste']:.3f} {row_mk[0]['hit_rate']:.3f}" if row_mk else "---"
        nw = f"{row_nw[0]['speedup']:.3f} {row_nw[0]['waste']:.3f} {row_nw[0]['hit_rate']:.3f}" if row_nw else "---"
        l16 = f"{row_l16[0]['speedup']:.3f} {row_l16[0]['waste']:.3f} {row_l16[0]['hit_rate']:.3f}" if row_l16 else "---"
        st = f"{row_st[0]['speedup']:.3f} {row_st[0]['waste']:.3f} {row_st[0]['hit_rate']:.3f}" if row_st else "---"
        print(f"  {wl_id:>{wl_col}s}  {mk:>16s}  {nw:>14s}  {l16:>14s}  {st:>14s}")
    
    print(f"\n  Saved: {out_dir / 'mixed_results.json'}")


if __name__ == '__main__':
    main()
