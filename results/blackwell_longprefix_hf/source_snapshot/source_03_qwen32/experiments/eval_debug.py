#!/usr/bin/env python3
"""Analyze fidelity results: empty generations and per-variant stats."""
import json

r = json.load(open('v10/fidelity_equiv_smoke3/all_results.json'))

# Per-variant stats
by_variant = {}
for item in r:
    key = (item['variant_a'], item['variant_b'])
    by_variant.setdefault(key, []).append(item)

print("=== Per-variant pair stats ===")
for (va, vb), items in sorted(by_variant.items()):
    empty_a = sum(1 for x in items if not x.get('exact_text','').strip())
    empty_b = sum(1 for x in items if not x.get('approx_text','').strip())
    print(f"  A={va} B={vb}: {len(items)} samples, empty_A={empty_a}, empty_B={empty_b}")

# Non-empty pairs only
non_empty = [x for x in r if x.get('exact_text','').strip() and x.get('approx_text','').strip()]
print(f"\n=== Non-empty pairs: {len(non_empty)}/{len(r)} ===")

# ROUGE-L on non-empty pairs
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
sum_f = 0
for idx, item in enumerate(non_empty):
    scores = scorer.score(item['exact_text'], item['approx_text'])
    f = scores['rougeL'].fmeasure
    sum_f += f
    if idx < 5 or f < 0.2:
        print(f"  [{item['variant_a']}->{item['variant_b']}] ROUGE={f:.4f}")
        print(f"    A: {item['exact_text'][:80]}")
        print(f"    B: {item['approx_text'][:80]}")

avg = sum_f / len(non_empty) if non_empty else 0
print(f"\nAvg ROUGE-L (non-empty only): {avg:.4f}")

# Exact match rate on non-empty
em = sum(1 for x in non_empty if x['exact_text'].strip().lower() == x['approx_text'].strip().lower())
print(f"Exact match (non-empty): {em}/{len(non_empty)} = {em/max(len(non_empty),1):.4f}")

# Length correlation
lens_a = [len(x['exact_text'].split()) for x in non_empty]
lens_b = [len(x['approx_text'].split()) for x in non_empty]
if lens_a and lens_b:
    corr = sum((a - sum(lens_a)/len(lens_a))*(b - sum(lens_b)/len(lens_b)) for a,b in zip(lens_a,lens_b))
    corr /= max((sum((a-sum(lens_a)/len(lens_a))**2 for a in lens_a) * sum((b-sum(lens_b)/len(lens_b))**2 for b in lens_b))**0.5, 0.001)
    print(f"Length correlation: {corr:.4f}")
    print(f"Mean len A: {sum(lens_a)/len(lens_a):.1f}, B: {sum(lens_b)/len(lens_b):.1f}")

# Template output analysis
print("\n=== Variant A (template 0) empty analysis ===")
empty_a_items = [x for x in r if not x.get('exact_text','').strip()]
print(f"Total empty A: {len(empty_a_items)}/{len(r)}")
for item in empty_a_items[:5]:
    print(f"  B text: {item['approx_text'][:80]}")
