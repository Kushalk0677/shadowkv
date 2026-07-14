#!/usr/bin/env python3
"""
Evaluate semantic fidelity results: compute ROUGE-L (content-only) and BERTScore
between exact-prefix and semantic-approximate generated text.

Strips format prefixes from all texts before scoring so the metric measures
content fidelity, not formatting coincidence.

Usage:
  python eval_fidelity.py --input results/all_results.json --output results/summary.json
"""

import argparse, json, re, sys
from pathlib import Path


# Patterns to strip from generated text before scoring
STRIP_PATTERNS = [
    re.compile(r'^(Response|Answer|Output|Instruction|Script|Task|Assistant'
               r'|System|Note|Explanation|Result|Summary|Synopsis'
               r'|Classification|Label|Category|Decision'
               r'|Solution|Advice|Suggestion|Tip'
               r'|Comment|Reply|Remark|Feedback'
               r'|Message|Chat|Dialogue|Conversation'
               r'|Review|Analysis|Assessment|Evaluation'
               r'|Description|Overview|Breakdown'
               r'|Details|Info|Information'
               r'|Step|Steps|Guide|Tutorial'
               r'|Example|Sample|Instance'
               r'|Option|Options|Choice|Choices'
            r'):\s*', re.IGNORECASE),
    re.compile(r'^\d+[\.\)]\s*'),  # numbered list
    re.compile(r'^-\s+'),          # bullet
    re.compile(r'^\n+'),           # leading newlines
]


def strip_format(text: str) -> str:
    """Remove format prefixes iteratively."""
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for pat in STRIP_PATTERNS:
            m = pat.match(text)
            if m:
                text = text[m.end():].strip()
                changed = True
                break
    return text


def compute_rouge_l(reference: str, hypothesis: str, strip: bool = True) -> dict:
    """Compute ROUGE-L score on content-only text."""
    if strip:
        reference = strip_format(reference)
        hypothesis = strip_format(hypothesis)
    if not reference or not hypothesis:
        return {'rougeL_precision': 0, 'rougeL_recall': 0, 'rougeL_fmeasure': 0}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            'rougeL_precision': round(scores['rougeL'].precision, 4),
            'rougeL_recall': round(scores['rougeL'].recall, 4),
            'rougeL_fmeasure': round(scores['rougeL'].fmeasure, 4),
        }
    except ImportError:
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        if not ref_tokens or not hyp_tokens:
            return {'rougeL_precision': 0, 'rougeL_recall': 0, 'rougeL_fmeasure': 0}
        overlap = ref_tokens & hyp_tokens
        precision = len(overlap) / len(hyp_tokens)
        recall = len(overlap) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {'rougeL_precision': round(precision, 4), 'rougeL_recall': round(recall, 4), 'rougeL_fmeasure': round(f1, 4)}


def compute_bertscore(references: list, hypotheses: list, model_type: str = 'distilbert-base-uncased') -> list:
    """Compute BERTScore for a batch. Returns None on error."""
    try:
        from bert_score import score as bscore
        P, R, F1 = bscore(hypotheses, references, model_type=model_type, verbose=False)
        return [{'precision': round(p.item(), 4), 'recall': round(r.item(), 4), 'f1': round(f.item(), 4)}
                for p, r, f in zip(P, R, F1)]
    except Exception:
        return None


def compute_exact_match(reference: str, hypothesis: str, strip: bool = True) -> int:
    if strip:
        reference = strip_format(reference)
        hypothesis = strip_format(hypothesis)
    return 1 if reference.strip().lower() == hypothesis.strip().lower() else 0


def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic fidelity')
    parser.add_argument('--input', required=True, help='Path to results JSON')
    parser.add_argument('--output', default='fidelity_summary.json', help='Output summary path')
    parser.add_argument('--bertscore_model', default='distilbert-base-uncased', help='BERTScore model')
    parser.add_argument('--no_strip', action='store_true', help='Disable format stripping')
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    print(f'Loaded {len(results)} results')
    strip_flag = not args.no_strip

    # Compute per-sample metrics
    for r in results:
        exact = r.get('exact_text', '')
        approx = r.get('approx_text', '')
        if strip_flag:
            exact_stripped = strip_format(exact)
            approx_stripped = strip_format(approx)
        else:
            exact_stripped = exact
            approx_stripped = approx

        r['rougeL'] = compute_rouge_l(exact, approx, strip=strip_flag)
        r['exact_match'] = compute_exact_match(exact, approx, strip=strip_flag)
        a_len = len(approx_stripped.split())
        e_len = max(len(exact_stripped.split()), 1)
        r['length_ratio'] = round(a_len / e_len, 4)
        r['exact_stripped'] = exact_stripped
        r['approx_stripped'] = approx_stripped

    # BERTScore
    references = [strip_format(r.get('exact_text', '')) for r in results] if strip_flag else [r.get('exact_text', '') for r in results]
    hypotheses = [strip_format(r.get('approx_text', '')) for r in results] if strip_flag else [r.get('approx_text', '') for r in results]
    bs = compute_bertscore(references, hypotheses, args.bertscore_model)
    if bs:
        for i, r in enumerate(results):
            r['bertscore'] = bs[i]
        avg_f1 = sum(b['f1'] for b in bs) / len(bs)
    else:
        avg_f1 = None
        print('BERTScore not available (continuing with ROUGE-L only).')

    # Aggregate by model and dataset
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r['model'], r['dataset'])].append(r)

    summary = {'per_sample': results, 'aggregate': {}}
    for (model, dataset), group in groups.items():
        avg_rouge = sum(r['rougeL']['rougeL_fmeasure'] for r in group) / len(group)
        avg_exact = sum(r['exact_match'] for r in group) / len(group)
        avg_len_ratio = sum(r['length_ratio'] for r in group) / len(group)
        entry = {
            'n': len(group),
            'avg_rougeL_f1': round(avg_rouge, 4),
            'exact_match_rate': round(avg_exact, 4),
            'avg_length_ratio': round(avg_len_ratio, 4),
        }
        if bs:
            bss = [r.get('bertscore', {}).get('f1', 0) for r in group]
            entry['avg_bertscore_f1'] = round(sum(bss) / len(bss), 4) if bss else None
        summary['aggregate'][f'{model}/{dataset}'] = entry

    # Overall
    all_rouge = [r['rougeL']['rougeL_fmeasure'] for r in results]
    all_exact = [r['exact_match'] for r in results]
    summary['overall'] = {
        'n': len(results),
        'avg_rougeL_f1': round(sum(all_rouge) / len(all_rouge), 4),
        'exact_match_rate': round(sum(all_exact) / len(all_exact), 4),
        'stripped': strip_flag,
    }
    if bs:
        summary['overall']['avg_bertscore_f1'] = round(avg_f1, 4)

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n=== RESULTS (stripped={strip_flag}) ===')
    print(f'Total samples: {len(results)}')
    print(f'Avg ROUGE-L F1: {summary["overall"]["avg_rougeL_f1"]}')
    print(f'Exact match rate: {summary["overall"]["exact_match_rate"]}')
    if bs:
        print(f'Avg BERTScore F1: {summary["overall"]["avg_bertscore_f1"]}')
    print(f'\nPer group:')
    for key, entry in sorted(summary['aggregate'].items()):
        bs_str = f' BERT={entry.get("avg_bertscore_f1", "N/A")}' if 'avg_bertscore_f1' in entry else ''
        print(f'  {key:30s} ROUGE={entry["avg_rougeL_f1"]:.4f} EM={entry["exact_match_rate"]:.4f}{bs_str}')
    print(f'\nSaved summary to {args.output}')


if __name__ == '__main__':
    main()
