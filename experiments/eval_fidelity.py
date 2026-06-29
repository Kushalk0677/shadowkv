#!/usr/bin/env python3
"""
Evaluate semantic fidelity results: compute ROUGE-L and BERTScore
between exact-prefix and semantic-approximate generated text.

Usage:
  python eval_fidelity.py --input fidelity_results/all_results.json --output fidelity_results/summary.json
"""

import argparse, json, sys
from pathlib import Path


def compute_rouge_l(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-L score. Returns precision, recall, f1."""
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
        # Fallback: simple token overlap
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
    """Compute BERTScore for a batch of reference/hypothesis pairs."""
    try:
        from bert_score import score as bscore
        P, R, F1 = bscore(hypotheses, references, model_type=model_type, verbose=False)
        return [{'precision': round(p.item(), 4), 'recall': round(r.item(), 4), 'f1': round(f.item(), 4)}
                for p, r, f in zip(P, R, F1)]
    except ImportError:
        return None


def compute_exact_match(reference: str, hypothesis: str) -> int:
    return 1 if reference.strip().lower() == hypothesis.strip().lower() else 0


def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic fidelity')
    parser.add_argument('--input', required=True, help='Path to results JSON from run_semantic_fidelity.py')
    parser.add_argument('--output', default='fidelity_summary.json', help='Output summary path')
    parser.add_argument('--bertscore_model', default='distilbert-base-uncased', help='BERTScore model')
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    print(f'Loaded {len(results)} results')

    # Compute per-sample metrics
    for r in results:
        exact = r.get('exact_text', '')
        approx = r.get('approx_text', '')
        if exact and approx:
            r['rougeL'] = compute_rouge_l(exact, approx)
            r['exact_match'] = compute_exact_match(exact, approx)
            r['length_ratio'] = round(len(approx.split()) / max(len(exact.split()), 1), 4)
        else:
            r['rougeL'] = {'rougeL_precision': 0, 'rougeL_recall': 0, 'rougeL_fmeasure': 0}
            r['exact_match'] = 0
            r['length_ratio'] = 0

    # Compute BERTScore in batch (needs GPU)
    references = [r.get('exact_text', '') for r in results]
    hypotheses = [r.get('approx_text', '') for r in results]
    bs = compute_bertscore(references, hypotheses, args.bertscore_model)
    if bs:
        for i, r in enumerate(results):
            r['bertscore'] = bs[i]
        avg_f1 = sum(b['f1'] for b in bs) / len(bs)
    else:
        avg_f1 = None
        print('BERTScore not available. Install: pip install bert-score')

    # Aggregate by model and dataset
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r['model'], r['dataset'])
        groups[key].append(r)

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
    }
    if bs:
        summary['overall']['avg_bertscore_f1'] = round(avg_f1, 4)

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n=== RESULTS ===')
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
