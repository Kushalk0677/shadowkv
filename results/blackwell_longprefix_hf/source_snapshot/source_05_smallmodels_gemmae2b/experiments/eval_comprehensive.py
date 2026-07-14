"""Full evaluation: prompt sensitivity (exact vs ref) + KV fidelity (ref vs reuse)."""
import json, sys, re
from collections import defaultdict

# Load our correct results
r = json.load(open('v10/fidelity_equiv_full/tinyllama_results.json'))

# Strip function for content-only comparison
STRIP_PAT = re.compile(
    r'^(Response:|Answer:|Output:|Instruction:|Script:|Task:|Assistant:|'
    r'System:|Note:|Explanation:|Result:|Summary:|Solution:|'
    r'Dialogue:|Conversation:|Messages:|Chat:|'
    r'Description:|Overview:|Info:|Information:)\s*', re.IGNORECASE
)

def clean(text):
    text = text.strip()
    while True:
        m = STRIP_PAT.match(text)
        if m:
            text = text[m.end():].strip()
        else:
            break
    return text

# Import ROUGE
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

rows = []
for x in r:
    exact_clean = clean(x['exact_text'])
    ref_clean = clean(x['ref_text'])
    reuse_clean = clean(x['reuse_text'])
    
    # KV fidelity (ref vs reuse)
    if ref_clean and reuse_clean:
        rr = scorer.score(ref_clean, reuse_clean)['rougeL'].fmeasure
    else:
        rr = 0
        if ref_clean or reuse_clean:
            rr = 1.0 if ref_clean == reuse_clean else 0.0
    
    # Prompt sensitivity (exact vs ref)
    if exact_clean and ref_clean:
        re = scorer.score(exact_clean, ref_clean)['rougeL'].fmeasure
    else:
        re = 0
    
    rows.append({
        'dataset': x['dataset'],
        'fidelity_rl': rr,
        'sensitivity_rl': re,
        'exact_match': ref_clean == reuse_clean,
        'exact_len': len(exact_clean.split()),
        'ref_len': len(ref_clean.split()),
        'reuse_len': len(reuse_clean.split()),
    })

# Aggregate
print("=" * 70)
print("COMPLETE FIDELITY EVALUATION — TinyLlama")
print("=" * 70)

# KV Fidelity
fid_all = [r['fidelity_rl'] for r in rows]
fid_exact = sum(1 for r in rows if r['exact_match'])
print(f"\nKV Fidelity (ref vs reuse):")
print(f"  Exact matches: {fid_exact}/{len(rows)} = {fid_exact/len(rows)*100:.1f}%")
print(f"  Avg ROUGE-L:   {sum(fid_all)/len(fid_all):.4f}")

# Prompt Sensitivity
sen_all = [r['sensitivity_rl'] for r in rows]
print(f"\nPrompt Sensitivity (exact vs ref):")
print(f"  Avg ROUGE-L:   {sum(sen_all)/len(sen_all):.4f}")
print(f"  Range:         {min(sen_all):.4f} - {max(sen_all):.4f}")

# By dataset
print(f"\n{'Dataset':<15} {'N':<4} {'Fidelity':<10} {'Sensitivity':<12} {'Exact Match':<12}")
print("-" * 55)
by_ds = defaultdict(list)
for r in rows:
    by_ds[r['dataset']].append(r)
for ds, items in sorted(by_ds.items()):
    n = len(items)
    f = sum(x['fidelity_rl'] for x in items) / n
    s = sum(x['sensitivity_rl'] for x in items) / n
    m = sum(1 for x in items if x['exact_match'])
    print(f"{ds:<15} {n:<4} {f:<10.4f} {s:<12.4f} {m}/{n}")

# Length analysis
print(f"\nGeneration Length:")
avg_exact = sum(r['exact_len'] for r in rows) / len(rows)
avg_ref = sum(r['ref_len'] for r in rows) / len(rows)
avg_reuse = sum(r['reuse_len'] for r in rows) / len(rows)
print(f"  Exact: {avg_exact:.1f} words")
print(f"  Ref:   {avg_ref:.1f} words")
print(f"  Reuse: {avg_reuse:.1f} words")
