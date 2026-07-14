"""Verify paper ROUGE-L values against all_results.json."""
import json, sys
from collections import defaultdict
from rouge_score import rouge_scorer

with open('v10/fidelity_results/fidelity_results/all_results.json') as f:
    data = json.load(f)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

sys.stdout.write(f'Total samples: {len(data)}\n\n')

# Paper claims vs actual
paper = {
    'gpt2': {'rouge': 0.876, 'exact': 79.2},
    'tinyllama': {'rouge': 0.966, 'exact': 96.8},
    'qwen25_15b': {'rouge': 0.200, 'exact': 0.8},
    'gemma2b': {'rouge': 0.974, 'exact': 95.1},
    'phi3mini': {'rouge': 0.931, 'exact': 83.7},
}

by_model = defaultdict(list)
for x in data:
    by_model[x['model']].append(x)

sys.stdout.write(f'{"Model":>15} {"N":>5} {"Exact%":>7} {"ROUGE":>7} {"PaperR":>7} {"R Match?":>9} {"PaperE%":>7} {"E Match?":>9}\n')
sys.stdout.write('-'*70 + '\n')

all_ok = True
for mk in ['gpt2', 'tinyllama', 'qwen25_15b', 'gemma2b', 'phi3mini']:
    items = by_model[mk]
    n = len(items)
    exact = sum(1 for x in items if x['ref_text'] == x['reuse_text'])
    exact_pct = round(exact / n * 100, 1)
    rs = [scorer.score(x['ref_text'], x['reuse_text'])['rougeL'].fmeasure
          for x in items if x['ref_text'] and x['reuse_text']]
    avg_rs = round(sum(rs) / len(rs), 3) if rs else 0.0

    pr = paper[mk]['rouge']
    pe = paper[mk]['exact']
    r_ok = abs(avg_rs - pr) < 0.01
    e_ok = abs(exact_pct - pe) < 0.5
    if not r_ok or not e_ok:
        all_ok = False

    sys.stdout.write(f'{mk:>15} {n:>5} {exact_pct:>6.1f}% {avg_rs:>7.3f} {pr:>7.3f} {"OK" if r_ok else "MISMATCH":>9} {pe:>6.1f}% {"OK" if e_ok else "MISMATCH":>9}\n')

sys.stdout.write(f'\nAll match: {all_ok}\n')
sys.stdout.flush()
