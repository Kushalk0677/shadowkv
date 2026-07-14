import json
r = json.load(open('v10/fidelity_equiv_final/tinyllama_results.json'))
print(f'Type: {type(r[0])}')
for k, v in r[0].items():
    print(f'  {k}: type={type(v).__name__}, val={repr(str(v)[:80])}')
