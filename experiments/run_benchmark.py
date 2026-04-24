from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import time

from proactive_kv_cache.datasets import list_datasets
from proactive_kv_cache.engines import (
    FrequencySpeculativeEngine,
    NativePrefixCachingEngine,
    NoCacheEngine,
    ReactivePrefixCacheEngine,
    ShadowKVEngine,
    StrictReactivePrefixCacheEngine,
    compare_named_runs,
    maybe_shutdown,
)
from proactive_kv_cache.models import load_backend
from proactive_kv_cache.policy import CostAwareSlackPolicy
from proactive_kv_cache.utils import set_seed
from proactive_kv_cache.workload import SYNTHETIC_VARIANTS, make_public_dataset_workload, make_synthetic_workload


MODEL_PRESETS = {
    'tiny': 'sshleifer/tiny-gpt2',
    'distilgpt2': 'distilgpt2',
    'gpt2': 'gpt2',
    'phi3mini': 'microsoft/Phi-3-mini-4k-instruct',
    'llama32_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'qwen25_15b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen25_3b': 'Qwen/Qwen2.5-3B-Instruct',
    'mistral7b': 'mistralai/Mistral-7B-Instruct-v0.3',
    'llama31_8b': 'meta-llama/Llama-3.1-8B-Instruct',
}


def resolve_model(model: str | None) -> str | None:
    if model is None:
        return None
    return MODEL_PRESETS.get(model, model)


def build_requests(args):
    if args.workload == 'synthetic':
        return make_synthetic_workload(variant=args.variant, n_requests=args.n_requests, seed=args.seed)
    if args.workload == 'public_dataset':
        return make_public_dataset_workload(dataset_name=args.dataset, split=args.dataset_split, n_requests=args.n_requests, seed=args.seed)
    raise ValueError(f'Unsupported workload: {args.workload}')


def maybe_sleep(current_idx: int, requests, simulate_arrivals: bool) -> None:
    if not simulate_arrivals or current_idx == 0:
        return
    delay = requests[current_idx].arrival_time - requests[current_idx - 1].arrival_time
    if delay > 0:
        time.sleep(min(delay, 0.2))


def build_engines(args, backend):
    engines = [NoCacheEngine(backend=backend, max_memory_mb=args.max_memory_mb)]

    if getattr(backend, 'supports_native_prefix_caching', False):
        engines.append(NativePrefixCachingEngine(backend=backend, max_memory_mb=args.max_memory_mb))

    if getattr(backend, 'supports_external_kv', True):
        engines.extend(
            [
                ReactivePrefixCacheEngine(backend=backend, max_memory_mb=args.max_memory_mb),
                StrictReactivePrefixCacheEngine(backend=backend, max_memory_mb=args.max_memory_mb),
            ]
        )
        if args.include_experimental:
            engines.extend(
                [
                    FrequencySpeculativeEngine(
                        backend=backend,
                        max_memory_mb=args.max_memory_mb,
                        speculative_k=args.speculative_k,
                        idle_threshold_ms=args.idle_threshold_ms,
                    ),
                    ShadowKVEngine(
                        backend=backend,
                        max_memory_mb=args.max_memory_mb,
                        speculative_k=args.speculative_k,
                        idle_threshold_ms=args.idle_threshold_ms,
                        policy=CostAwareSlackPolicy(
                            min_frequency=0.20 if backend.device.startswith('cuda') else 0.24,
                            token_benefit_ms=1.2 if backend.device.startswith('cuda') else 1.05,
                            speculation_penalty_ms=5.0 if backend.device.startswith('cuda') else 7.0,
                            min_prefix_len=24 if backend.device.startswith('cuda') else 12,
                            preferred_prefix_len=96 if backend.device.startswith('cuda') else 48,
                            max_prefix_len=192 if backend.device.startswith('cuda') else 128,
                            min_observations=3 if backend.device.startswith('cuda') else 4,
                            min_expected_net_ms=8.0 if backend.device.startswith('cuda') else 12.0,
                            benefit_cost_ratio=1.25 if backend.device.startswith('cuda') else 1.35,
                            gpu_bonus_ms=8.0 if backend.device.startswith('cuda') else 2.0,
                            max_admissions_per_idle=1 if backend.device.startswith('cuda') else 2,
                            min_recent_support=0.05 if backend.device.startswith('cuda') else 0.04,
                        ),
                        enable_gpu_tier=args.device.startswith('cuda'),
                    ),
                ]
            )
    return engines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['fake', 'hf', 'vllm'], default='fake')
    parser.add_argument('--model', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='auto')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--disable_native_prefix_caching', action='store_true')
    parser.add_argument('--include_experimental', action='store_true')
    parser.add_argument('--workload', choices=['synthetic', 'public_dataset'], default='synthetic')
    parser.add_argument('--variant', choices=sorted(SYNTHETIC_VARIANTS.keys()), default='high_skew')
    parser.add_argument('--dataset', choices=list_datasets(), default='daily_dialog')
    parser.add_argument('--dataset_split', default=None)
    parser.add_argument('--n_requests', type=int, default=60)
    parser.add_argument('--simulate_arrivals', action='store_true')
    parser.add_argument('--max_memory_mb', type=int, default=64)
    parser.add_argument('--speculative_k', type=int, default=2)
    parser.add_argument('--idle_threshold_ms', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()

    set_seed(args.seed)
    backend = load_backend(
        args.backend,
        model_name=resolve_model(args.model),
        device=args.device,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=not args.disable_native_prefix_caching,
    )
    requests = build_requests(args)
    engines = build_engines(args, backend)

    for engine in engines:
        for idx, req in enumerate(requests):
            maybe_sleep(idx, requests, args.simulate_arrivals)
            tokens = backend.tokenize(req.prompt)
            engine.serve_tokens(req.request_id, tokens)
        maybe_shutdown(engine)

    summary = compare_named_runs(engines)
    summary['config'] = vars(args)
    summary['config']['resolved_model'] = resolve_model(args.model)
    summary['capabilities'] = {
        'supports_external_kv': getattr(backend, 'supports_external_kv', False),
        'supports_native_prefix_caching': getattr(backend, 'supports_native_prefix_caching', False),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = (resolve_model(args.model) or args.model or 'default').replace('/', '_').replace(':', '_')
    workload_slug = args.dataset if args.workload == 'public_dataset' else args.variant
    name = f"benchmark_{args.backend}_{model_slug}_{args.workload}_{workload_slug}_{args.device.replace(':', '_')}.json"
    out_file = output_dir / name
    out_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved to {out_file}')


if __name__ == '__main__':
    main()
