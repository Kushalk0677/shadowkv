#!/usr/bin/env python3
"""Quick evaluation of fidelity results with content-only ROUGE-L."""
import json, re, sys
from rouge_score import rouge_scorer

def strip_format(text):
    """Remove format prefixes."""
    text = text.strip()
    pat = re.compile(
        r'^(Response:|Answer:|Output:|Instruction:|Script:|Task:|Assistant:|'
        r'System:|Note:|Explanation:|Result:|Summary:|'
        r'Solution:|Advice:|Suggestion:|Comment:|Reply:|'
        r'Dialogue:|Conversation:|Messages:|Chat:|'
        r'Description:|Overview:|Info:|Information:|'
        r'Step |Steps |Guide:|Example:)\s*', re.IGNORECASE
    )
    while True:
        m = pat.match(text)
        if m:
            text = text[m.end():].strip()
        else:
            break
    return text

data = json.load(open(sys.argv[1]))
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

print(f"Total samples: {len(data)}\n")

non_empty = [x for x in data if x.get('exact_text','').strip() and x.get('approx_text','').strip()]
raw_scores = []
clean_scores = []

for item in data:
    a, b = item.get('exact_text',''), item.get('approx_text','')
    a_strip, b_strip = strip_format(a), strip_format(b)
    item['exact_clean'] = a_strip
    item['approx_clean'] = b_strip
    
    # Raw score (no stripping)
    if a.strip() and b.strip():
        rs = scorer.score(a, b)['rougeL'].fmeasure
    else:
        rs = 0.0
    raw_scores.append(rs)
    
    # Clean score (stripped)
    if a_strip and b_strip:
        cs = scorer.score(a_strip, b_strip)['rougeL'].fmeasure
    else:
        cs = 0.0
    clean_scores.append(cs)
    
    item['rougeL_raw'] = round(rs, 4)
    item['rougeL_clean'] = round(cs, 4)

# Show per-sample results
print(f"{'Sample':>6} {'Model':>12} {'Dataset':>12} {'Var':>5} {'Raw':>6} {'Clean':>6}  {'A_text':<40} {'B_text':<40}")
print("-"*140)
for i, item in enumerate(data):
    a50 = strip_format(item['exact_text'])[:38]
    b50 = strip_format(item['approx_text'])[:38]
    print(f"{item['sample_idx']:>6} {item['model']:>12} {item['dataset']:>12} "
          f"{item['variant_a']}->{item['variant_b']:>3} "
          f"{item['rougeL_raw']:>6.4f} {item['rougeL_clean']:>6.4f}  "
          f"{a50:<40} {b50:<40}")

# Summary by dataset
print(f"\n{'Dataset':>12} {'N':>4} {'Raw Avg':>8} {'Clean Avg':>8}")
print("-"*40)
from collections import defaultdict
by_ds = defaultdict(list)
for i, item in enumerate(data):
    by_ds[item['dataset']].append(i)

for ds, idxs in sorted(by_ds.items()):
    r_avg = sum(raw_scores[i] for i in idxs) / len(idxs)
    c_avg = sum(clean_scores[i] for i in idxs) / len(idxs)
    print(f"{ds:>12} {len(idxs):>4} {r_avg:>8.4f} {c_avg:>8.4f}")

all_raw = sum(raw_scores) / len(raw_scores)
all_clean = sum(clean_scores) / len(clean_scores)
print(f"\nOverall: N={len(data)}, Raw ROUGE-L: {all_raw:.4f}, Clean ROUGE-L: {all_clean:.4f}")
