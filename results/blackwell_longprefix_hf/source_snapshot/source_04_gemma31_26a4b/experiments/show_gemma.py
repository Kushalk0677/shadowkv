import json, sys
r = json.load(open('v10/fidelity_equiv_gemma/gemma2b_results.json'))
for item in r:
    same = item['ref_text'] == item['reuse_text']
    sys.stdout.write(f'Gemma 2B — shared_ratio={item["shared_ratio"]}\n')
    sys.stdout.write(f'ref==reuse: {same}\n')
    sys.stdout.write(f'  ref:   {item["ref_text"][:80]}\n')
    sys.stdout.write(f'  reuse: {item["reuse_text"][:80]}\n')
    sys.stdout.write(f'  exact: {item["exact_text"][:80]}\n')
sys.stdout.flush()
