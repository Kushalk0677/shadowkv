"""Evaluate all 5 model results from Colab run."""
import json, sys
from collections import defaultdict

results = []
for mk in ['gpt2_results', 'tinyllama_results', 'qwen25_15b_results', 'gemma2b_results', 'phi3mini_results']:
    with open(f'v10/fidelity_results/fidelity_results/{mk}.json') as f:
        results.extend(json.load(f))

print(f'Total samples: {len(results)}')
print()

matches = sum(1 for x in results if x['ref_text'] == x['reuse_text'])
print(f'KV Fidelity: {matches}/{len(results)} = {matches/len(results)*100:.1f}%')
print()

# By model
print(f'{"Model":>15} {"Samples":>8} {"KV Match":>10} {"Rate":>8}')
print('-'*45)
by_model = defaultdict(list)
for x in results:
    by_model[x['model']].append(x)
for mk in ['gpt2','tinyllama','qwen25_15b','gemma2b','phi3mini']:
    items = by_model[mk]
    m = sum(1 for x in items if x['ref_text'] == x['reuse_text'])
    n = len(items)
    print(f'{mk:>15} {n:>8} {m:>8}/{n:<3} {m/n*100:>7.1f}%')

# By dataset
print()
print(f'{"Dataset":>18} {"Samples":>8} {"KV Match":>10} {"Rate":>8}')
print('-'*48)
by_ds = defaultdict(list)
for x in results:
    by_ds[x['dataset']].append(x)
total_samp = 0
total_match = 0
for dk in ['samsum','xsum','cnn_dailymail','ag_news','banking77','alpaca_eval','dolly','daily_dialog','oasst1','ultrachat']:
    items = by_ds.get(dk, [])
    if items:
        m = sum(1 for x in items if x['ref_text'] == x['reuse_text'])
        n = len(items)
        total_samp += n
        total_match += m
        print(f'{dk:>18} {n:>8} {m:>8}/{n:<3} {m/n*100:>7.1f}%')
print(f'{"TOTAL":>18} {total_samp:>8} {total_match:>8}/{total_samp:<3} {total_match/total_samp*100:>7.1f}%')
