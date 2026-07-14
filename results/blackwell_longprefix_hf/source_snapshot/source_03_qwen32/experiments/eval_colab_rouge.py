"""Full evaluation with ROUGE-L from Colab GPU results."""
import json, sys, re
from collections import defaultdict
from rouge_score import rouge_scorer

results = []
for mk in ['gpt2_results', 'tinyllama_results', 'qwen25_15b_results', 'gemma2b_results', 'phi3mini_results']:
    with open(f'v10/fidelity_results/fidelity_results/{mk}.json') as f:
        results.extend(json.load(f))

print(f'Total samples: {len(results)}')
print()

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# KV Fidelity: ref vs reuse (exact match + ROUGE-L)
exact_matches = 0
rouge_scores = []
for x in results:
    if x['ref_text'] == x['reuse_text']:
        exact_matches += 1
    # ROUGE-L (content-only, strip common prefixes)
    ref = x['ref_text'].strip()
    reuse = x['reuse_text'].strip()
    if ref and reuse:
        s = scorer.score(ref, reuse)['rougeL'].fmeasure
        rouge_scores.append(s)

print(f'KV Fidelity:')
print(f'  Exact string match: {exact_matches}/{len(results)} = {exact_matches/len(results)*100:.1f}%')
print(f'  Avg ROUGE-L (ref vs reuse): {sum(rouge_scores)/len(rouge_scores):.4f}')
print()

# By model
print(f'{"Model":>15} {"N":>5} {"Exact%":>8} {"ROUGE-L":>8}')
print('-'*40)
by_model = defaultdict(list)
for x in results:
    by_model[x['model']].append(x)
for mk in ['gpt2','tinyllama','qwen25_15b','gemma2b','phi3mini']:
    items = by_model[mk]
    n = len(items)
    em = sum(1 for x in items if x['ref_text'] == x['reuse_text'])
    rs = [scorer.score(x['ref_text'], x['reuse_text'])['rougeL'].fmeasure 
          for x in items if x['ref_text'] and x['reuse_text']]
    avg_rs = sum(rs)/len(rs) if rs else 0
    print(f'{mk:>15} {n:>5} {em/n*100:>7.1f}% {avg_rs:>8.4f}')

# Prompt sensitivity (exact vs ref)
print()
print(f'Prompt Sensitivity (exact vs ref):')
by_model2 = defaultdict(list)
for x in results:
    by_model2[x['model']].append(x)
for mk in ['gpt2','tinyllama','qwen25_15b','gemma2b','phi3mini']:
    items = by_model2[mk]
    rs = [scorer.score(x['exact_text'], x['ref_text'])['rougeL'].fmeasure 
          for x in items if x['exact_text'] and x['ref_text']]
    avg_rs = sum(rs)/len(rs) if rs else 0
    print(f'  {mk:>15}: {avg_rs:.4f}')

# By dataset (aggregate across models)
print()
print(f'{"Dataset":>18} {"N":>5} {"Exact%":>8} {"ROUGE-L":>8}')
print('-'*42)
by_ds = defaultdict(list)
for x in results:
    by_ds[x['dataset']].append(x)
for dk in ['samsum','xsum','cnn_dailymail','ag_news','banking77','alpaca_eval','dolly','daily_dialog','oasst1','ultrachat']:
    items = by_ds.get(dk, [])
    if items:
        n = len(items)
        em = sum(1 for x in items if x['ref_text'] == x['reuse_text'])
        rs = [scorer.score(x['ref_text'], x['reuse_text'])['rougeL'].fmeasure 
              for x in items if x['ref_text'] and x['reuse_text']]
        avg_rs = sum(rs)/len(rs) if rs else 0
        print(f'{dk:>18} {n:>5} {em/n*100:>7.1f}% {avg_rs:>8.4f}')
