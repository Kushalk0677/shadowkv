"""Sensitivity sweep for the ~40 unvalidated constants in UtilityModel.

Tests UtilityModel.admission() directly with synthetic AdmissionEvent objects,
bypassing the engine fast path so parameter changes actually affect decisions.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\kusha\Documents\GitHub\shadowkv")
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.chdir(ROOT)

from proactive_kv_cache.config_loader import CONFIG
from proactive_kv_cache.utility import UtilityModel, AdmissionEvent
from proactive_kv_cache.semantic import token_entropy

# Generate a deterministic token sequence
def make_tokens(n: int, base: int = 100) -> tuple:
    return tuple(range(base, base + n))


SWEEPS = [
    {
        'name': 'lambda (risk_aversion)',
        'path': 'policy.utility.risk_aversion',
        'default': 0.15,
        'values': [0.0, 0.05, 0.15, 0.30, 0.50],
        'fmt': '.2f',
    },
    {
        'name': 'semantic_threshold',
        'path': 'policy.utility.semantic_threshold',
        'default': 0.58,
        'values': [0.40, 0.50, 0.58, 0.70, 0.85],
        'fmt': '.2f',
    },
    {
        'name': 'util_min_ms',
        'path': 'policy.utility.util_min_ms',
        'default': 0.0,
        'values': [0.0, 2.0, 5.0, 10.0, 20.0],
        'fmt': '.1f',
    },
    {
        'name': 'semantic_prompt_benefit_discount',
        'path': 'policy.utility.semantic_prompt_benefit_discount',
        'default': 0.72,
        'values': [0.40, 0.55, 0.72, 0.85, 1.0],
        'fmt': '.2f',
    },
    {
        'name': 'semantic_default_benefit_discount',
        'path': 'policy.utility.semantic_default_benefit_discount',
        'default': 0.55,
        'values': [0.30, 0.45, 0.55, 0.72, 0.90],
        'fmt': '.2f',
    },
    {
        'name': 'semantic_prompt_waste_floor',
        'path': 'policy.utility.semantic_prompt_waste_floor',
        'default': 0.03,
        'values': [0.01, 0.03, 0.05, 0.10, 0.20],
        'fmt': '.2f',
    },
    {
        'name': 'lambda_m (memory weight)',
        'path': 'policy.utility.lambda_m_weight',
        'default': 0.02,
        'values': [0.0, 0.01, 0.02, 0.05, 0.10],
        'fmt': '.3f',
    },
    {
        'name': 'ewma_alpha',
        'path': 'policy.health.ewma_alpha',
        'default': 0.20,
        'values': [0.05, 0.10, 0.20, 0.40, 0.60],
        'fmt': '.2f',
    },
    {
        'name': 'entropy_penalty_weight',
        'path': 'policy.signals.entropy_penalty_weight',
        'default': 0.35,
        'values': [0.0, 0.15, 0.35, 0.55, 0.75],
        'fmt': '.2f',
    },
    {
        'name': 'exact_confidence_base',
        'path': 'policy.utility.exact_confidence_base',
        'default': 0.45,
        'values': [0.20, 0.35, 0.45, 0.60, 0.80],
        'fmt': '.2f',
    },
]


def make_event(
    prompt_mode='raw',
    exact_match_len=0,
    semantic_similarity=0.0,
    semantic_prefix_len=0,
    shared_prefix_hint=None,
    memory_bytes=0,
    tier=None,
    ewma_hit_rate=0.5,
    ewma_waste_ratio=0.15,
    full_ms_per_token=0.6,
    reuse_overhead_ms=2.0,
    observation_count=5,
    n_tokens=128,
):
    tokens = make_tokens(n_tokens)
    return AdmissionEvent(
        tokens=tokens,
        exact_match_len=exact_match_len,
        semantic_similarity=semantic_similarity,
        semantic_prefix_len=semantic_prefix_len,
        shared_prefix_hint=shared_prefix_hint,
        full_ms_per_token=full_ms_per_token,
        reuse_overhead_ms=reuse_overhead_ms,
        ewma_hit_rate=ewma_hit_rate,
        ewma_waste_ratio=ewma_waste_ratio,
        max_layer_reuse_ratio=0.50,
        min_utility_ms=0.0,
        semantic_threshold=0.58,
        metadata={'prompt_mode': prompt_mode},
        tier=tier,
        memory_bytes=memory_bytes,
        observation_count=observation_count,
    )


# Test scenarios
SCENARIOS = [
    {
        'name': 'Exact match, 64 tok, GPU tier',
        'event': make_event(exact_match_len=64, tier='gpu', memory_bytes=64*1024),
    },
    {
        'name': 'Exact match, 32 tok, CPU tier',
        'event': make_event(exact_match_len=32, tier='cpu', memory_bytes=32*1024),
    },
    {
        'name': 'Exact match, small (16 tok)',
        'event': make_event(exact_match_len=16, tier='gpu', memory_bytes=16*1024),
    },
    {
        'name': 'Exact match, low hit rate (0.1)',
        'event': make_event(exact_match_len=64, tier='gpu', ewma_hit_rate=0.1, 
                            ewma_waste_ratio=0.3),
    },
    {
        'name': 'Semantic, high sim (0.85), prompt mode',
        'event': make_event(prompt_mode='semantic', semantic_similarity=0.85, 
                            semantic_prefix_len=64, shared_prefix_hint=48),
    },
    {
        'name': 'Semantic, medium sim (0.60)',
        'event': make_event(prompt_mode='semantic', semantic_similarity=0.60,
                            semantic_prefix_len=48, shared_prefix_hint=32),
    },
    {
        'name': 'Semantic, low sim (0.40)',
        'event': make_event(prompt_mode='semantic', semantic_similarity=0.40,
                            semantic_prefix_len=32, shared_prefix_hint=16),
    },
    {
        'name': 'Semantic, high waste (0.40)',
        'event': make_event(prompt_mode='semantic', semantic_similarity=0.70,
                            semantic_prefix_len=48, shared_prefix_hint=32,
                            ewma_waste_ratio=0.4),
    },
    {
        'name': 'No match (bypass baseline)',
        'event': make_event(),
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    sweeps = SWEEPS
    if args.quick:
        for s in sweeps:
            v = s['values']
            if len(v) > 3:
                s['values'] = [v[0], v[len(v)//2], v[-1]]

    total_runs = sum(len(s['values']) for s in sweeps) * len(SCENARIOS)
    print(f"Sensitivity: {len(sweeps)} params x {len(SCENARIOS)} scenarios = {total_runs} calls")
    print()

    all_results = []

    for sweep in sweeps:
        name = sweep['name']
        path = sweep['path']
        default = sweep['default']
        fmt = sweep['fmt']

        print(f"\n{'='*70}")
        print(f"  {name}  (default={default:{fmt}})")
        print(f"{'='*70}")

        # For each value, run all scenarios
        value_results = []
        for value in sweep['values']:
            CONFIG.update({path: float(value)})
            CONFIG._cached_get_results.clear()
            model = UtilityModel(CONFIG)

            is_default = abs(value - default) < 1e-9
            label = f"{value:{fmt}}"
            note = "(default)" if is_default else ""
            
            row = {'value': value, 'is_default': is_default}
            for sc in SCENARIOS:
                breakdown = model.admission(sc['event'])
                row[sc['name']] = {
                    'strategy': breakdown.strategy,
                    'net_utility_ms': round(breakdown.net_utility_ms, 2),
                    'benefit_ms': round(breakdown.benefit_ms, 2),
                    'cost_ms': round(breakdown.cost_ms, 2),
                    'waste_ms': round(breakdown.waste_ms, 2),
                    'confidence': round(breakdown.confidence, 3),
                    'reason': breakdown.reason,
                }
            value_results.append(row)

        # Print table
        sc_names = [s['name'][:20] for s in SCENARIOS]
        header = f"  {'Value':>8}"
        for sn in sc_names:
            header += f"  {sn:>12}"
        header += "  Notes"
        print(header)
        
        sep = f"  {'-'*8}"
        for sn in sc_names:
            sep += f"  {'-'*12}"
        print(sep)

        for row in value_results:
            val_str = f"{row['value']:{fmt}}"
            line = f"  {val_str:>8}"
            for sc in SCENARIOS:
                r = row[sc['name']]
                # Show: strategy(utility)
                strat = r['strategy'][:4]
                util = r['net_utility_ms']
                line += f"  {strat}({util:>6.1f})"
            note = "  (default)" if row['is_default'] else ""
            print(line + note)

        all_results.append({
            'param': name,
            'path': path,
            'default': default,
            'values': value_results,
            'scenarios': [s['name'] for s in SCENARIOS],
        })

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SENSITIVITY SUMMARY (max delta in net_utility_ms across values)")
    print(f"{'='*70}\n")

    header = f"  {'Parameter':>30s}"
    for sc in SCENARIOS:
        header += f"  {sc['name'][:24]:>24s}"
    print(header)
    print(f"  {'-'*30}  " + "  ".join(['-'*24 for _ in SCENARIOS]))

    for sweep_data in all_results:
        name = sweep_data['param']
        line = f"  {name:>30s}"
        for si, sc in enumerate(SCENARIOS):
            utils = [v[sc['name']]['net_utility_ms'] for v in sweep_data['values']]
            delta = max(utils) - min(utils)
            strats = [v[sc['name']]['strategy'] for v in sweep_data['values']]
            flipped = len(set(strats)) > 1
            marker = " *" if flipped else "  "
            line += f"  {delta:>6.1f}{marker:>3s}"
        print(line)

    print()
    print("  * = parameter change flips the admission decision (admit vs bypass)")

    # Save
    out_dir = Path(r"C:\shadowkv\results\sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'utility_sweep_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'sweeps': sweeps,
            'scenarios': [s['name'] for s in SCENARIOS],
            'results': all_results,
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
