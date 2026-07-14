from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proactive_kv_cache.policy_learning import load_feature_rows, learn_shadowkv_plus_thresholds, rows_to_csv


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize ShadowKV/ShadowKV++ benchmark JSONs and result zips.')
    parser.add_argument('paths', nargs='*', default=[str(ROOT / 'results')])
    parser.add_argument('--csv', default=str(ROOT / 'results' / 'shadowkv_result_summary.csv'))
    parser.add_argument('--markdown', default=str(ROOT / 'results' / 'shadowkv_result_summary.md'))
    parser.add_argument('--policy-json', default=str(ROOT / 'results' / 'shadowkv_plus_learned_policy.json'))
    args = parser.parse_args()

    rows = load_feature_rows(args.paths)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv).write_text(rows_to_csv(rows))

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r.engine, r.prompt_mode)].append(r)

    lines = [
        '# ShadowKV result summary',
        '',
        f'Parsed runs: {len(rows)}',
        '',
        '| engine | prompt_mode | n | mean speedup | mean hit rate | mean waste | mean reuse density |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for (engine, prompt_mode), rs in sorted(grouped.items()):
        lines.append(
            f'| {engine} | {prompt_mode} | {len(rs)} | '
            f'{_mean([r.speedup_vs_no_cache for r in rs]):.3f} | '
            f'{_mean([r.cache_hit_rate for r in rs]):.3f} | '
            f'{_mean([r.wasted_compute_ratio for r in rs]):.3f} | '
            f'{_mean([r.reuse_density for r in rs]):.3f} |'
        )

    policy = learn_shadowkv_plus_thresholds(rows)
    Path(args.policy_json).write_text(json.dumps(policy, indent=2))
    lines.extend([
        '',
        '## Learned deployment gate',
        '',
        'The dependency-free learner searches conservative thresholds for enabling ShadowKV++ on future workload families.',
        '',
        '```json',
        json.dumps(policy, indent=2),
        '```',
        '',
        '## Interpretation guidance',
        '',
        '- Treat fake-backend runs as regression/smoke evidence only.',
        '- Treat HF/vLLM repeated seeded runs as performance evidence.',
        '- ShadowKV++ metrics to report: policy net utility, semantic match rate, semantic partial hits, layer reuse events, and waste ratio.',
    ])
    Path(args.markdown).write_text('\n'.join(lines) + '\n')

    print(f'Wrote {args.csv}')
    print(f'Wrote {args.markdown}')
    print(f'Wrote {args.policy_json}')


if __name__ == '__main__':
    main()
