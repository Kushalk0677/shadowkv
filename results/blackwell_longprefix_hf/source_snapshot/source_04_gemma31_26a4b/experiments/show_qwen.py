import json, sys
r = json.load(open('v10/fidelity_equiv_qwen/qwen25_15b_results.json'))
for item in r:
    same = item['ref_text'] == item['reuse_text']
    sys.stdout.write(f'Qwen 2.5 1.5B — shared_ratio=0.75\n')
    sys.stdout.write(f'ref==reuse: {same}\n')
    sys.stdout.write(f'  ref:   {item["ref_text"][:80]}\n')
    sys.stdout.write(f'  reuse: {item["reuse_text"][:80]}\n')
    sys.stdout.write(f'  exact: {item["exact_text"][:80]}\n')
sys.stdout.flush()
