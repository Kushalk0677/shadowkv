"""Evaluate control experiment: ratio=0.0 should show 100% match."""
import json

with open('v10/fidelity_equiv_control2/tinyllama_results.json') as f:
    data = json.load(f)

print(f'Total samples: {len(data)}')
matches = sum(1 for x in data if x['ref_text'] == x['reuse_text'])
print(f'ratio=0.0 ref==reuse: {matches}/{len(data)} = {matches/len(data)*100:.1f}%')
print()

# Show each sample
from collections import Counter
by_ds = Counter()
for x in data:
    by_ds[x['dataset']] += 1
    r_match = x['ref_text'] == x['reuse_text']
    print(f'  {x["dataset"]:>15} sample {x["sample_idx"]:>2}: ref==reuse={r_match}')
    print(f'    ref:   {x["ref_text"][:60]}')
    print(f'    reuse: {x["reuse_text"][:60]}')
