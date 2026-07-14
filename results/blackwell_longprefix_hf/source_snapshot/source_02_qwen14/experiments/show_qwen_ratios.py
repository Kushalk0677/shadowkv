import json, sys
r = json.load(open('v10/fidelity_equiv_qwen_ratios/qwen25_15b_results.json'))
for item in r:
    same = item['ref_text'] == item['reuse_text']
    sys.stdout.write(f'ratio={item["shared_ratio"]} ref==reuse: {same}\n')
    sys.stdout.write(f'  ref:   {item["ref_text"][:70]}\n')
    sys.stdout.write(f'  reuse: {item["reuse_text"][:70]}\n')
    sys.stdout.write(f'  exact: {item["exact_text"][:70]}\n\n')
sys.stdout.flush()
