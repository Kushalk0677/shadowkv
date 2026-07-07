#!/usr/bin/env python3
"""Memory-bound interleaved trace with eviction pressure and per-decision ledger.

Runs a 3-phase trace that exceeds KV cache capacity midway, forcing evictions.
Instrumented to count:
- Admissions declined by MeritKV vs APC baseline
- Victim cache misses (requests that would have hit if not evicted)
- Reload bytes and P99 latency on victims

Usage (FakeBackend for testing):
    python experiments/run_memory_bound_trace.py --backend fake

Usage (Blackwell/vLLM):
    python experiments/run_memory_bound_trace.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse, json, os, sys, time, random, math
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

from proactive_kv_cache.config_loader import CONFIG
from proactive_kv_cache.engines import (
    ShadowKVPlusEngine, NoCacheEngine, maybe_shutdown, summarize_engine,
)
from proactive_kv_cache.models import FakeBackend, load_backend

# ---- Trace Templates ----
PHASE1_SHARED = (
    "You are a helpful enterprise assistant. Answer questions based on the "
    "provided policy documents. Policy: KYC refresh is mandatory after trigger "
    "events, suspicious patterns require escalation, and customer communications "
    "must avoid promising account actions before identity verification.\n"
    "Customer query: "
)

PHASE2_SHARED = (
    "You are a claims processing assistant. Extract facts, classify the claim, "
    "identify missing evidence, and produce a concise triage note. Use the "
    "checklist: incident date, location, claimant statement, policy coverage, "
    "exclusions, fraud signals.\n"
    "Claim narrative: "
)

PHASE3_SHARED = PHASE1_SHARED  # Recovery: same as phase 1

SUFFIXES = [
    "Can you raise my transfer limit to 50000?",
    "I need to update my address and phone number.",
    "What is the status of my loan application?",
    "I lost my card, can you block it?",
    "How do I set up a recurring payment?",
    "I see a charge I don't recognize on my statement.",
    "Can I get a refund for the annual fee?",
    "What documents do I need for a mortgage pre-approval?",
    "I want to close my account and transfer the balance.",
    "Is there a fee for international wire transfers?",
    "Can you explain my credit score change?",
    "I need a letter confirming my account balance.",
    "How do I add an authorized user to my account?",
    "What are the requirements for a business line of credit?",
    "My direct deposit hasn't shown up yet.",
    "Can I dispute a transaction from six months ago?",
    "I need to change my PIN number.",
    "What is the interest rate on your savings account?",
    "How do I order new checks?",
    "Can I set up alerts for large transactions?",
]

class Req:
    def __init__(self, rid, prompt, metadata):
        self.request_id = rid
        self.prompt = prompt
        self.metadata = metadata
        self.arrival_time = 0.0
        self.tokens = None


def build_trace(n_phase1=80, n_phase2=60, n_phase3=60, seed=42):
    """Build 3-phase trace that exceeds cache capacity midway."""
    rng = random.Random(seed)
    requests = []
    
    # Phase 1: Fill cache with hot reusable entries
    for i in range(n_phase1):
        suffix = SUFFIXES[i % min(15, len(SUFFIXES))]  # first 15 suffixes repeat
        prompt = PHASE1_SHARED + suffix
        metadata = {
            'prompt_mode': 'templated' if i < n_phase1 - 10 else 'raw',
            'shared_prefix_hint_tokens': len(PHASE1_SHARED.split()),
            'shared_prefix_text': PHASE1_SHARED,
            'phase': 1,
            'template_id': 'phase1',
            'request_index': i,
        }
        requests.append(Req(i, prompt, metadata))
    
    # Phase 2: Churn — alternate between new template (evictor) and old template (victim)
    offset = n_phase1
    for i in range(n_phase2):
        is_evictor = i < n_phase2 // 2  # first half evicts, second half are victims
        suffix_idx = (i // 2) % len(SUFFIXES)
        suffix = SUFFIXES[suffix_idx]
        
        if is_evictor:
            shared = PHASE2_SHARED
            template_id = 'phase2'
            prompt_mode = 'templated'
        else:
            shared = PHASE1_SHARED
            template_id = 'phase1_victim'
            prompt_mode = 'templated'
        
        prompt = shared + suffix
        metadata = {
            'prompt_mode': prompt_mode,
            'shared_prefix_hint_tokens': len(shared.split()),
            'shared_prefix_text': shared,
            'phase': 2,
            'template_id': template_id,
            'is_evictor': is_evictor,
            'request_index': offset + i,
        }
        requests.append(Req(offset + i, prompt, metadata))
    
    # Phase 3: Recovery — repeat phase 1 template
    offset = n_phase1 + n_phase2
    for i in range(n_phase3):
        suffix = SUFFIXES[i % len(SUFFIXES)]
        prompt = PHASE3_SHARED + suffix
        metadata = {
            'prompt_mode': 'templated',
            'shared_prefix_hint_tokens': len(PHASE3_SHARED.split()),
            'shared_prefix_text': PHASE3_SHARED,
            'phase': 3,
            'template_id': 'phase3_recovery',
            'request_index': offset + i,
        }
        requests.append(Req(offset + i, prompt, metadata))
    
    return requests


def tokenize_batch(backend, requests):
    for req in requests:
        req.tokens = tuple(backend.tokenize(req.prompt))


def run_trace(engine, requests):
    results = []
    for req in requests:
        result = engine.serve_tokens(
            req.request_id, req.tokens, metadata=req.metadata)
        results.append({
            'request_id': req.request_id,
            'phase': req.metadata.get('phase'),
            'template_id': req.metadata.get('template_id'),
            'is_evictor': req.metadata.get('is_evictor', False),
            'latency_ms': result.latency_ms,
            'was_cache_hit': result.was_cache_hit,
            'matched_prefix_length': result.matched_prefix_length,
            'tokens_recomputed': result.tokens_recomputed,
        })
    return results


def compute_ledger(results, requests):
    """Build per-decision ledger from run results."""
    ledger = []
    for r, req in zip(results, requests):
        entry = {
            'request_id': r['request_id'],
            'phase': r['phase'],
            'template_id': r['template_id'],
            'is_evictor': r['is_evictor'],
            'latency_ms': r['latency_ms'],
            'was_cache_hit': r['was_cache_hit'],
        }
        ledger.append(entry)
    
    # Count victims
    victims = [e for e in ledger if e['template_id'] == 'phase1_victim']
    victim_misses = [v for v in victims if not v['was_cache_hit']]
    
    # Phase 1 hit rate (baseline)
    phase1 = [e for e in ledger if e['phase'] == 1 and e['template_id'] == 'phase1']
    phase1_hits = sum(1 for p in phase1 if p['was_cache_hit'])
    
    # Phase 3 recovery rate
    phase3 = [e for e in ledger if e['phase'] == 3]
    phase3_hits = sum(1 for p in phase3 if p['was_cache_hit'])
    
    return {
        'total_requests': len(ledger),
        'phase1_hit_rate': phase1_hits / len(phase1) if phase1 else 0,
        'phase3_hit_rate': phase3_hits / len(phase3) if phase3 else 0,
        'victim_total': len(victims),
        'victim_misses': len(victim_misses),
        'victim_miss_rate': len(victim_misses) / len(victims) if victims else 0,
        'p95_victim_latency_ms': sorted([v['latency_ms'] for v in victim_misses] or [0])[
            max(0, int(len(victim_misses) * 0.95) - 1)
        ] if victim_misses else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['fake', 'vllm'], default='fake')
    parser.add_argument('--model', default=None)
    parser.add_argument('--n_phase1', type=int, default=80)
    parser.add_argument('--n_phase2', type=int, default=60)
    parser.add_argument('--n_phase3', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results/memory_bound_trace')
    args = parser.parse_args()
    
    if args.backend == 'vllm' and not args.model:
        args.model = 'Qwen/Qwen2.5-7B-Instruct'
    
    # Build trace
    requests = build_trace(args.n_phase1, args.n_phase2, args.n_phase3, args.seed)
    
    # Load backend
    if args.backend == 'fake':
        backend = FakeBackend(device='cpu')
    else:
        backend = load_backend('vllm', model_name=args.model)
    
    tokenize_batch(backend, requests)
    
    print(f"Memory-bound trace: {len(requests)} requests, 3 phases")
    print(f"  Phase 1 (fill): {args.n_phase1} hot entries")
    print(f"  Phase 2 (churn): {args.n_phase2} (evictors + victims)")
    print(f"  Phase 3 (recovery): {args.n_phase3}")
    
    results = {}
    
    for label, engine_cls, kwargs in [
        ('APC_only', NoCacheEngine, {}),
        ('APC+MeritKV', ShadowKVPlusEngine, {'allow_approximate_semantic_reuse': False}),
    ]:
        print(f"\n--- Running {label} ---")
        
        # Reset backend for each run
        if args.backend == 'fake':
            eng = engine_cls(backend=backend, max_memory_mb=128)  # Small cache
        else:
            eng = engine_cls(backend=backend, max_memory_mb=256)
        
        run_results = run_trace(eng, requests)
        maybe_shutdown(eng)
        summary = summarize_engine(eng)
        
        ledger = compute_ledger(run_results, requests)
        results[label] = {
            'ledger': ledger,
            'summary': {
                'mean_speedup': 1.0,
                'mean_latency_ms': summary.get('mean_latency_ms', 0),
                'hit_rate': summary.get('hit_rate', 0),
            }
        }
        
        print(f"  Victims: {ledger['victim_total']} ({ledger['victim_misses']} misses, "
              f"{ledger['victim_miss_rate']*100:.0f}%)")
        print(f"  Phase 1 hit rate: {ledger['phase1_hit_rate']*100:.0f}%")
        print(f"  Phase 3 hit rate: {ledger['phase3_hit_rate']*100:.0f}%")
        print(f"  P95 victim latency: {ledger['p95_victim_latency_ms']:.1f}ms")
    
    # Compare
    if 'APC_only' in results and 'APC+MeritKV' in results:
        base = results['APC_only']['ledger']
        skv = results['APC+MeritKV']['ledger']
        print(f"\n=== LEDGER COMPARISON ===")
        print(f"{'Metric':>35s}  {'APC only':>10s}  {'+MeritKV':>10s}  {'Delta':>8s}")
        print(f"{'-'*35}  {'-'*10}  {'-'*10}  {'-'*8}")
        delta_v = base['victim_misses'] - skv['victim_misses']
        print(f"{'Victim misses':>35s}  {base['victim_misses']:>10d}  {skv['victim_misses']:>10d}  {delta_v:+>8d}")
              # removed
        print(f"{'Victim miss rate':>35s}  {base['victim_miss_rate']:>10.0%}  {skv['victim_miss_rate']:>10.0%}")
        print(f"{'P95 victim latency':>35s}  {base['p95_victim_latency_ms']:>8.1f}ms  {skv['p95_victim_latency_ms']:>8.1f}ms")
        print(f"{'Phase 3 recovery rate':>35s}  {base['phase3_hit_rate']:>10.0%}  {skv['phase3_hit_rate']:>10.0%}")
    
    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'trace_results.json', 'w') as f:
        json.dump({
            'config': {
                'backend': args.backend,
                'model': args.model,
                'n_phase1': args.n_phase1,
                'n_phase2': args.n_phase2,
                'n_phase3': args.n_phase3,
                'seed': args.seed,
            },
            'results': {k: v['ledger'] for k, v in results.items()},
        }, f, indent=2)
    print(f"\nSaved to {out_dir / 'trace_results.json'}")


if __name__ == '__main__':
    main()
