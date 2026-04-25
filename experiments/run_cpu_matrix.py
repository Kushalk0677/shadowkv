from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


MODEL_PRESETS = {
    'tiny': 'sshleifer/tiny-gpt2',
    'distilgpt2': 'distilgpt2',
}

DATASETS = [
    'ag_news',
    'alpaca_eval',
    'banking77',
    'cnn_dailymail',
    'daily_dialog',
    'dolly',
    'oasst1',
    'samsum',
    'ultrachat',
    'xsum',
]


def result_name(model: str, dataset: str) -> str:
    resolved = MODEL_PRESETS.get(model, model)
    model_slug = resolved.replace('/', '_').replace(':', '_')
    return f'benchmark_hf_{model_slug}_public_dataset_{dataset}_cpu.json'


def result_matches_config(path: Path, model: str, dataset: str, n_requests: int) -> bool:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    config = data.get('config', {})
    return (
        config.get('resolved_model') == MODEL_PRESETS.get(model, model)
        and config.get('dataset') == dataset
        and int(config.get('n_requests', -1)) == int(n_requests)
        and config.get('device') == 'cpu'
    )


def summarize_result(path: Path) -> dict:
    data = json.loads(path.read_text())
    engines = {}
    for name, summary in data.items():
        if isinstance(summary, dict) and 'mean_latency_ms' in summary:
            engines[name] = {
                'mean_latency_ms': summary['mean_latency_ms'],
                'p95_latency_ms': summary['p95_latency_ms'],
                'hit_rate': summary['hit_rate'],
                'reuse_success_rate': summary.get('reuse_success_rate', 0.0),
                'wasted_compute_ratio': summary['wasted_compute_ratio'],
                'speedup_vs_no_cache_mean': summary.get('speedup_vs_no_cache_mean', 0.0),
            }
    winner = min(engines.items(), key=lambda item: item[1]['mean_latency_ms'])[0] if engines else None
    return {'winner': winner, 'engines': engines}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='results/cpu_matrix_20260422')
    parser.add_argument('--n_requests', type=int, default=16)
    parser.add_argument('--timeout_seconds', type=int, default=600)
    parser.add_argument('--models', nargs='+', default=['tiny', 'distilgpt2'])
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    status_path = out_dir / 'matrix_status.json'
    summary_path = out_dir / 'matrix_summary.json'

    status_by_key: dict[str, dict] = {}
    if status_path.exists():
        try:
            loaded = json.loads(status_path.read_text())
            if isinstance(loaded, list):
                for item in loaded:
                    if isinstance(item, dict) and 'model' in item and 'dataset' in item:
                        status_by_key[f"{item['model']}::{item['dataset']}"] = item
        except json.JSONDecodeError:
            status_by_key = {}

    for model in args.models:
        for dataset in args.datasets:
            result_path = out_dir / result_name(model, dataset)
            entry = {
                'model': model,
                'dataset': dataset,
                'result_file': str(result_path.relative_to(root)),
                'status': 'pending',
                'elapsed_seconds': 0.0,
            }
            key = f'{model}::{dataset}'

            if result_path.exists() and not args.force and result_matches_config(result_path, model, dataset, args.n_requests):
                entry['status'] = 'skipped_existing'
                entry['summary'] = summarize_result(result_path)
                status_by_key[key] = entry
                status_path.write_text(json.dumps(list(status_by_key.values()), indent=2))
                print(f'SKIP existing model={model} dataset={dataset}', flush=True)
                continue

            command = [
                sys.executable,
                str(root / 'experiments' / 'run_benchmark.py'),
                '--backend',
                'hf',
                '--model',
                model,
                '--device',
                'cpu',
                '--workload',
                'public_dataset',
                '--dataset',
                dataset,
                '--n_requests',
                str(args.n_requests),
                '--max_memory_mb',
                '64',
                '--speculative_k',
                '2',
                '--idle_threshold_ms',
                '20',
                '--include_experimental',
                '--output_dir',
                str(out_dir),
            ]
            log_path = logs_dir / f'{model}_{dataset}.log'
            print(f'RUN model={model} dataset={dataset}', flush=True)
            started = time.perf_counter()
            if log_path.exists():
                log_path.unlink()
            try:
                with log_path.open('w', encoding='utf-8') as log_file:
                    log_file.write('COMMAND:\n' + ' '.join(command) + '\n\n')
                    log_file.flush()
                    completed = subprocess.run(
                        command,
                        cwd=root,
                        text=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout_seconds,
                    )
                elapsed = time.perf_counter() - started
                entry['elapsed_seconds'] = round(elapsed, 2)
                entry['exit_code'] = completed.returncode
                if completed.returncode == 0 and result_path.exists():
                    entry['status'] = 'completed'
                    entry['summary'] = summarize_result(result_path)
                else:
                    entry['status'] = 'failed'
                    entry['log_file'] = str(log_path.relative_to(root))
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - started
                stdout = exc.stdout.decode('utf-8', errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
                stderr = exc.stderr.decode('utf-8', errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
                with log_path.open('a', encoding='utf-8') as log_file:
                    log_file.write('\nTIMEOUT\n')
                    if stdout:
                        log_file.write('\nSTDOUT:\n' + stdout)
                    if stderr:
                        log_file.write('\nSTDERR:\n' + stderr)
                entry['status'] = 'timeout'
                entry['elapsed_seconds'] = round(elapsed, 2)
                entry['log_file'] = str(log_path.relative_to(root))

            status_by_key[key] = entry
            status_path.write_text(json.dumps(list(status_by_key.values()), indent=2))

    current_keys = {f'{model}::{dataset}' for model in args.models for dataset in args.datasets}
    status = [status_by_key[key] for key in current_keys if key in status_by_key]
    completed = [
        item for item in status
        if item.get('status') in {'completed', 'skipped_existing'} and item.get('summary')
    ]
    failed = [item for item in status if item.get('status') in {'failed', 'timeout'}]
    summary = {
        'completed_or_existing': len(completed),
        'failed_or_timeout': len(failed),
        'items': completed,
        'failures': failed,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({'completed_or_existing': len(completed), 'failed_or_timeout': len(failed)}, indent=2))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
