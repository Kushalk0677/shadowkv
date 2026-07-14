"""Deep analysis of Qwen's KV reuse failure on GPU."""
import json

with open('v10/fidelity_results/fidelity_results/qwen25_15b_results.json') as f:
    data = json.load(f)

print(f'Qwen samples: {len(data)}')
print()

# Check a few samples
for i in range(min(5, len(data))):
    item = data[i]
    ref = item['ref_text'][:80]
    reuse = item['reuse_text'][:80]
    exact = item['exact_text'][:80]
    same = item['ref_text'] == item['reuse_text']
    print(f'--- Sample {i} (ratio={item["shared_ratio"]}) ref==reuse: {same} ---')
    print(f'  ref:   {ref}')
    print(f'  reuse: {reuse}')
    print(f'  exact: {exact}')
    print()

# Check first token divergence
print('=== First few tokens of ref vs reuse (sample 0) ===')
item = data[0]
ref_tokens = item['ref_text'].split()[:10]
reuse_tokens = item['reuse_text'].split()[:10]
for i, (r, ru) in enumerate(zip(ref_tokens, reuse_tokens)):
    match = '✓' if r == ru else '✗'
    print(f'  Token {i}: ref={r:>15} reuse={ru:>15} {match}')

# Count empty outputs
empty_ref = sum(1 for x in data if not x['ref_text'].strip())
empty_reuse = sum(1 for x in data if not x['reuse_text'].strip())
print(f'\nEmpty ref_text: {empty_ref}/{len(data)}')
print(f'Empty reuse_text: {empty_reuse}/{len(data)}')

# Check shared token ratios
from collections import Counter
ratios = Counter(x['shared_ratio'] for x in data)
print(f'\nRatios: {dict(ratios)}')
