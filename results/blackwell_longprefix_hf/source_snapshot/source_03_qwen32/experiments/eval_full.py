"""Evaluate full results: count ref==reuse matches and compute ROUGE-L."""
import json, sys
from collections import defaultdict

r = json.load(open('v10/fidelity_equiv_full/tinyllama_results.json'))

# Count matches
total = len(r)
matches = sum(1 for x in r if x['ref_text'] == x['reuse_text'])
sys.stdout.write(f'\n=== TinyLlama Fidelity Results ===\n')
sys.stdout.write(f'Total samples: {total}\n')
sys.stdout.write(f'ref==reuse (exact match): {matches}/{total} = {matches/total*100:.1f}%\n\n')

# By dataset
by_ds = defaultdict(list)
for x in r:
    by_ds[x['dataset']].append(x)
for ds, items in sorted(by_ds.items()):
    n = len(items)
    m = sum(1 for x in items if x['ref_text'] == x['reuse_text'])
    avg_shared = sum(x['shared_tokens'] for x in items) / n
    avg_total = sum(x['total_tokens_orig'] for x in items) / n
    ratio = avg_shared / avg_total * 100 if avg_total > 0 else 0
    sys.stdout.write(f'  {ds}: {m}/{n} matched, shared={ratio:.0f}%\n')

# Compare ref vs exact (prompt sensitivity)
import re
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def strip(t):
    t = t.strip()
    return re.sub(r'^(Response:|Answer:|Output:|Instruction:|Script:)\s*', '', t, flags=re.I).strip()

sys.stdout.write(f'\n=== ROUGE-L Scores ===\n')
ref_reuse_scores = []
ref_exact_scores = []
for x in r:
    if x['ref_text'] and x['reuse_text']:
        rr = scorer.score(strip(x['ref_text']), strip(x['reuse_text']))
        ref_reuse_scores.append(rr['rougeL'].fmeasure)
    if x['ref_text'] and x['exact_text']:
        re = scorer.score(strip(x['ref_text']), strip(x['exact_text']))
        ref_exact_scores.append(re['rougeL'].fmeasure)

if ref_reuse_scores:
    avg_rr = sum(ref_reuse_scores) / len(ref_reuse_scores)
    sys.stdout.write(f'ref vs reuse (KV fidelity):   {avg_rr:.4f} (n={len(ref_reuse_scores)})\n')
if ref_exact_scores:
    avg_re = sum(ref_exact_scores) / len(ref_exact_scores)
    sys.stdout.write(f'ref vs exact (prompt sensitivity): {avg_re:.4f} (n={len(ref_exact_scores)})\n')

sys.stdout.write(f'\nNote: ref==reuse measures KV cache reuse fidelity.\n')
sys.stdout.write(f'      ref vs exact measures prompt sensitivity (expect lower).\n')
sys.stdout.flush()
