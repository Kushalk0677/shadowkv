import json, sys
r = json.load(open('v10/fidelity_equiv_final2/tinyllama_results.json'))
for i, item in enumerate(r):
    same = item['ref_text'] == item['reuse_text']
    sys.stdout.write(f'[{i}] shared={item["shared_tokens"]}/{item["total_tokens_orig"]} ref==reuse: {same}\n')
    sys.stdout.write(f'  ref:   {item["ref_text"][:60]}\n')
    sys.stdout.write(f'  reuse: {item["reuse_text"][:60]}\n')
    sys.stdout.write(f'  exact: {item["exact_text"][:60]}\n\n')
sys.stdout.flush()
