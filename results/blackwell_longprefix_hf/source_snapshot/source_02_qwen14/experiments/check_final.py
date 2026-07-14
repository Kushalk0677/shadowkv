import json
from rouge_score import rouge_scorer

r = json.load(open('v10/fidelity_equiv_final/tinyllama_results.json'))
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

print('Results: ref vs reuse comparison\n')

for i, item in enumerate(r):
    ref = item['ref_text']
    reuse = item['reuse_text']
    exact = item['exact_text']
    same = ref == reuse
    
    # ROUGE scores
    rr = scorer.score(ref, reuse)['rougeL'].fmeasure if ref and reuse else 0
    re = scorer.score(ref, exact)['rougeL'].fmeasure if ref and exact else 0
    
    print(f'[{i}] shared={item["shared_tokens"]}/{item["total_tokens_orig"]} '
          f'ref==reuse: {same}  ROUGE(ref,reuse)={rr:.4f}  ROUGE(ref,exact)={re:.4f}')
    print(f'  ref:   {ref[:60]}')
    print(f'  reuse: {reuse[:60]}')
    print(f'  exact: {exact[:60]}')
    print()
