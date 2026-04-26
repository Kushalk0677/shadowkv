from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in root.rglob('policy_trace.jsonl'):
        with path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row['_trace_file'] = str(path)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Build request-level policy dataset from policy_trace.jsonl files.')
    parser.add_argument('results_root', type=Path)
    parser.add_argument('--output_jsonl', type=Path, default=Path('policy_dataset.jsonl'))
    args = parser.parse_args()

    rows = load_rows(args.results_root)
    # Add optional within-run no_cache baseline labels when available.
    by_case: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get('_trace_file'), row.get('model'), row.get('workload'), row.get('dataset'),
            row.get('variant'), row.get('prompt_mode'), row.get('seed'), row.get('request_id')
        )
        if row.get('engine') == 'no_cache':
            by_case[key] = row

    enriched = []
    for row in rows:
        key = (
            row.get('_trace_file'), row.get('model'), row.get('workload'), row.get('dataset'),
            row.get('variant'), row.get('prompt_mode'), row.get('seed'), row.get('request_id')
        )
        baseline = by_case.get(key)
        out = dict(row)
        if baseline and row.get('engine') != 'no_cache':
            baseline_latency = float(baseline.get('latency_ms') or 0.0)
            latency = float(row.get('latency_ms') or 0.0)
            out['baseline_no_cache_latency_ms'] = baseline_latency
            out['actual_utility_vs_no_cache_ms'] = baseline_latency - latency
            out['label_should_reuse_or_policy_positive'] = 1 if (baseline_latency - latency) > 0 else 0
        enriched.append(out)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open('w', encoding='utf-8') as fh:
        for row in enriched:
            fh.write(json.dumps(row, sort_keys=True) + '\n')
    print(f'Loaded {len(rows)} trace rows')
    print(f'Wrote {len(enriched)} policy dataset rows to {args.output_jsonl}')


if __name__ == '__main__':
    main()
