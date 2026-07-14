"""Analyze exact sample counts per model per dataset from Colab results."""
import json
from collections import defaultdict

results = []
for mk in ['gpt2', 'tinyllama', 'qwen25_15b', 'gemma2b', 'phi3mini']:
    with open(f'v10/fidelity_results/fidelity_results/{mk}_results.json') as f:
        results.extend(json.load(f))

# Count per model per dataset
counts = defaultdict(lambda: defaultdict(int))
for x in results:
    counts[x['model']][x['dataset']] += 1

datasets = ['samsum','xsum','cnn_dailymail','ag_news','banking77','alpaca_eval','dolly','daily_dialog','oasst1','ultrachat']
models = ['gpt2','tinyllama','qwen25_15b','gemma2b','phi3mini']

print(f'{"Model":>12}  ', end='')
for dk in datasets:
    print(f'{dk:>14}  ', end='')
print(f'{"Total":>8}')
print('-' * 120)

for mk in models:
    print(f'{mk:>12}  ', end='')
    total = 0
    for dk in datasets:
        c = counts[mk][dk]
        total += c
        print(f'{c:>14}  ', end='')
    print(f'{total:>8}')

# Grand total
print(f'{"TOTAL":>12}  ', end='')
gt = 0
for dk in datasets:
    c = sum(counts[mk][dk] for mk in models)
    gt += c
    print(f'{c:>14}  ', end='')
print(f'{gt:>8}')
