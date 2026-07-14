#!/usr/bin/env python3
"""Analyze fidelity with prefix stripping for content-only ROUGE-L."""
import json, re
from rouge_score import rouge_scorer

r = json.load(open('v10/fidelity_equiv_smoke3/all_results.json'))

def strip_format(text):
    """Remove formatting prefixes that differ between templates."""
    # Remove leading whitespace/newlines
    text = text.strip()
    # Remove common format prefixes
    prefixes = [
        r'^Response:\s*', r'^Answer:\s*', r'^Output:\s*', 
        r'^Instruction:\s*', r'^Script:\s*', r'^Task:\s*',
        r'^Assistant:\s*', r'^System:\s*',
        r'^\d+\.\s*',  # numbered list prefix
    ]
    for p in prefixes:
        text = re.sub(p, '', text)
    return text.strip()

# Show before/after stripping for first 5 samples
print("=== Prefix stripping examples ===")
for item in r[:5]:
    print(f"  A raw: {repr(item['exact_text'][:60])}")
    print(f"  A clean: {repr(strip_format(item['exact_text'])[:60])}")
    print(f"  B raw: {repr(item['approx_text'][:60])}")
    print(f"  B clean: {repr(strip_format(item['approx_text'])[:60])}")
    print()

# Compute ROUGE-L with stripping
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

non_empty = [x for x in r if x.get('exact_text','').strip() and x.get('approx_text','').strip()]
print(f"=== Non-empty pairs: {len(non_empty)}/{len(r)} ===")

raw_total = 0
clean_total = 0
for item in non_empty:
    raw_scores = scorer.score(item['exact_text'], item['approx_text'])
    clean_scores = scorer.score(strip_format(item['exact_text']), strip_format(item['approx_text']))
    raw_total += raw_scores['rougeL'].fmeasure
    clean_total += clean_scores['rougeL'].fmeasure

avg_raw = raw_total / len(non_empty)
avg_clean = clean_total / len(non_empty)
print(f"Avg ROUGE-L (raw text):       {avg_raw:.4f}")
print(f"Avg ROUGE-L (stripped text):  {avg_clean:.4f}")

# Also show variant-specific breakdown
by_variant = {}
for item in non_empty:
    key = (item['variant_a'], item['variant_b'])
    by_variant.setdefault(key, []).append(item)

print("\n=== Per-variant ROUGE-L (stripped) ===")
for (va, vb), items in sorted(by_variant.items()):
    total = 0
    for item in items:
        scores = scorer.score(strip_format(item['exact_text']), strip_format(item['approx_text']))
        total += scores['rougeL'].fmeasure
    avg = total / len(items)
    print(f"  A={va} B={vb}: n={len(items)}, ROUGE-L={avg:.4f}")

# Show best and worst cases
print("\n=== Best 5 (stripped) ===")
clean_scores_list = []
for item in non_empty:
    scores = scorer.score(strip_format(item['exact_text']), strip_format(item['approx_text']))
    clean_scores_list.append((scores['rougeL'].fmeasure, item))
clean_scores_list.sort(key=lambda x: -x[0])
for f, item in clean_scores_list[:5]:
    print(f"  ROUGE={f:.4f} A={item['variant_a']}->B={item['variant_b']}")
    print(f"    A: {strip_format(item['exact_text'])[:60]}")
    print(f"    B: {strip_format(item['approx_text'])[:60]}")

print("\n=== Worst 5 (stripped) ===")
for f, item in clean_scores_list[-5:]:
    print(f"  ROUGE={f:.4f} A={item['variant_a']}->B={item['variant_b']}")
    print(f"    A: {strip_format(item['exact_text'])[:60]}")
    print(f"    B: {strip_format(item['approx_text'])[:60]}")
