import json
r = json.load(open('v10/fidelity_equiv_v7/tinyllama_results.json'))
for i, item in enumerate(r):
    same = item['ref_text'] == item['reuse_text']
    same_exact = item['exact_text'] == item['reuse_text']
    print(f'[{i}] ref==reuse: {same}  exact==reuse: {same_exact}')
    print(f'  ref:   {item["ref_text"][:80]}')
    print(f'  reuse: {item["reuse_text"][:80]}')
    print(f'  exact: {item["exact_text"][:80]}')
    print()
