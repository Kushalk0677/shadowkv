from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proactive_kv_cache.config_loader import CONFIG
from proactive_kv_cache.models import load_backend


MODEL_PRESETS = {
    'gpt2': 'gpt2',
    'phi3mini': 'microsoft/Phi-3-mini-4k-instruct',
    'qwen25_15b': 'Qwen/Qwen2.5-1.5B-Instruct',
}


def _fit_line(xs: list[int], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        slope = ys[0] / max(xs[0], 1) if xs else 0.60
        return float(slope), 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return float(mean_y / max(mean_x, 1)), 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return float(max(slope, 0.0)), float(max(intercept, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile model/GPU prefill and KV costs into config.yaml.')
    parser.add_argument('--backend', choices=['fake', 'hf', 'vllm'], default='hf')
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--config', default=str(ROOT / 'config' / 'config.yaml'))
    parser.add_argument('--lengths', default='8,16,32,48,64')
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--memory_bandwidth_gbps', type=float, default=None)
    args = parser.parse_args()

    CONFIG.load(args.config)
    model = MODEL_PRESETS.get(args.model, args.model)
    backend = load_backend(
        args.backend,
        model_name=model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    lengths = [int(x) for x in args.lengths.split(',') if x.strip()]
    max_len = max(lengths)
    prompt = ('profiling token ' * (max_len * 4)).strip()
    tokens = backend.tokenize(prompt)
    if len(tokens) < max_len:
        raise RuntimeError(f'Tokenizer produced only {len(tokens)} tokens, need {max_len}.')

    warmup = tuple(tokens[:max_len])
    backend.prefill(warmup)
    measured_lengths: list[int] = []
    latencies: list[float] = []
    for length in lengths:
        sample = tuple(tokens[:length])
        timings: list[float] = []
        for _ in range(max(args.repeats, 1)):
            out = backend.prefill(sample)
            timings.append(float(out.latency_ms))
        measured_lengths.append(length)
        latencies.append(statistics.median(timings))

    beta, delta = _fit_line(measured_lengths, latencies)
    kv_mb_per_token = max(float(backend.estimate_kv_cache_bytes(max_len)) / (max_len * 1024.0 * 1024.0), 0.0005)
    updates = {
        'profile_id': f'{args.backend}:{model}:{args.device}:{int(time.time())}',
        'hardware.beta_prefill_ms_per_token': beta,
        'hardware.reuse_fixed_overhead_ms': delta,
        'hardware.kv_mb_per_token': kv_mb_per_token,
        'hardware.profiled_model': model,
        'hardware.profiled_device': args.device,
        'hardware.profiled_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    if args.memory_bandwidth_gbps is not None:
        updates['hardware.memory_bandwidth_gbps'] = float(args.memory_bandwidth_gbps)
    CONFIG.update(updates)
    CONFIG.write(args.config)
    print({
        'lengths': measured_lengths,
        'latencies_ms': latencies,
        'beta_prefill_ms_per_token': beta,
        'reuse_fixed_overhead_ms': delta,
        'kv_mb_per_token': kv_mb_per_token,
        'config': str(args.config),
    })


if __name__ == '__main__':
    main()
