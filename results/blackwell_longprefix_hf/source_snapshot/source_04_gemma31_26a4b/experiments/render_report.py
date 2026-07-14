from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def render_markdown(result: dict, source_name: str) -> str:
    config = result.get('config', {})
    engines = [(name, summary) for name, summary in result.items() if isinstance(summary, dict) and 'mean_latency_ms' in summary]
    engines.sort(key=lambda item: item[1]['mean_latency_ms'])
    def escape_cell(value: object) -> str:
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = []
    lines.append(f"# Benchmark report: {source_name}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for key in ('backend', 'resolved_model', 'device', 'workload', 'dataset', 'variant', 'n_requests'):
        if key in config:
            lines.append(f"- **{key}**: {config[key]}")
    lines.append("")
    lines.append("## Ranked results")
    lines.append("")
    if not engines:
        lines.append("_No engine summaries were found in this result file._")
        lines.append("")
        return "\n".join(lines)
    lines.append("| Engine | Mean latency (ms) | P95 latency (ms) | Speedup vs baseline | Hit rate | Reuse success rate | Cache active at end | Auto-disabled reason |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|---|")
    for name, s in engines:
        lines.append(
            f"| {escape_cell(name)} | {s['mean_latency_ms']:.2f} | {s['p95_latency_ms']:.2f} | {s.get('speedup_vs_no_cache_mean', 0.0):.3f} | {s.get('hit_rate', 0.0):.3f} | {s.get('reuse_success_rate', 0.0):.3f} | {escape_cell(str(s.get('cache_active_final', True)))} | {escape_cell(s.get('auto_disabled_reason') or '')} |"
        )
    lines.append("")
    best_name, best_summary = engines[0]
    baseline = result.get('no_cache', {})
    if best_name == 'no_cache':
        insight = "No cache-based engine beat the no-cache baseline on this workload."
    else:
        delta = baseline.get('mean_latency_ms', 0.0) - best_summary['mean_latency_ms']
        insight = f"Best engine: **{best_name}** with {delta:.2f} ms lower mean latency than `no_cache`."
    lines.append("## Key takeaway")
    lines.append("")
    lines.append(insight)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('result_json')
    parser.add_argument('--output_md', default=None)
    args = parser.parse_args()

    path = Path(args.result_json)
    result = load_result(path)
    md = render_markdown(result, path.name)
    output = Path(args.output_md) if args.output_md else path.with_suffix('.report.md')
    output.write_text(md)
    print(md)
    print(f"Saved report to {output}")


if __name__ == '__main__':
    main()
