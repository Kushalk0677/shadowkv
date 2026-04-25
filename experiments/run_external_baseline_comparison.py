from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# vLLM requires spawn when CUDA may already be initialized in notebook-style
# environments such as Colab.
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import run_benchmark as benchmark
from proactive_kv_cache.datasets import list_datasets, list_prompt_modes
from proactive_kv_cache.engines import NativePrefixCachingEngine, ShadowKVEngine, maybe_shutdown, summarize_engine
from proactive_kv_cache.models import load_backend
from proactive_kv_cache.policy import CostAwareSlackPolicy
from proactive_kv_cache.utils import set_seed
from proactive_kv_cache.workload import SYNTHETIC_VARIANTS


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
    except Exception:
        return


def _add_shared_prefix_metadata(backend, tokens, metadata: dict, cache: dict[str, int]) -> dict:
    shared_prefix_text = metadata.get('shared_prefix_text')
    if not shared_prefix_text:
        return metadata
    hint_len = cache.get(shared_prefix_text)
    if hint_len is None:
        hint_len = len(backend.tokenize(shared_prefix_text))
        cache[shared_prefix_text] = hint_len
    metadata['shared_prefix_hint_tokens'] = min(int(hint_len), len(tokens))
    return metadata


def _run_engine(engine_key: str, backend, engine, requests, simulate_arrivals: bool, max_arrival_sleep_ms: float) -> dict:
    shared_prefix_token_cache: dict[str, int] = {}
    try:
        for idx, req in enumerate(requests):
            benchmark.maybe_sleep(idx, requests, simulate_arrivals, max_arrival_sleep_ms)
            tokens = backend.tokenize(req.prompt)
            metadata = dict(req.metadata or {})
            metadata['arrival_time'] = req.arrival_time
            metadata = _add_shared_prefix_metadata(backend, tokens, metadata, shared_prefix_token_cache)
            engine.serve_tokens(req.request_id, tokens, metadata=metadata)
        maybe_shutdown(engine)
        summary = summarize_engine(engine)
        summary['backend'] = getattr(backend, 'backend_name', 'unknown')
        summary['engine'] = engine_key
        return summary
    finally:
        shutdown = getattr(backend, 'shutdown', None)
        if callable(shutdown):
            shutdown()
        del engine
        del backend
        _release_cuda_memory()


def _run_baseline(engine_key: str, build_fn, requests, simulate_arrivals: bool, max_arrival_sleep_ms: float):
    try:
        backend, engine = build_fn()
        capabilities = {
            'supports_external_kv': getattr(backend, 'supports_external_kv', False),
            'supports_native_prefix_caching': getattr(backend, 'supports_native_prefix_caching', False),
            'backend_name': getattr(backend, 'backend_name', 'unknown'),
        }
        summary = _run_engine(
            engine_key,
            backend,
            engine,
            requests,
            simulate_arrivals,
            max_arrival_sleep_ms,
        )
        return summary, capabilities
    except Exception as exc:
        _release_cuda_memory()
        return {
            'backend': engine_key,
            'engine': engine_key,
            'error': f'{exc.__class__.__name__}: {exc}',
            'status': 'failed',
        }, {
            'supports_external_kv': None,
            'supports_native_prefix_caching': None,
            'backend_name': engine_key,
        }


def _build_shadow_kv(args):
    backend = load_backend(
        'hf',
        model_name=benchmark.resolve_model(args.model),
        device=args.device,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        trust_remote_code=args.trust_remote_code,
    )
    calibration_backend = load_backend(
        'hf',
        model_name=benchmark.resolve_model(args.model),
        device=args.device,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        shadowkv_policy_kwargs, shadowkv_policy_calibration = benchmark._build_shadowkv_policy_kwargs(
            calibration_backend,
            prompt_mode=args.resolved_prompt_mode,
        )
    finally:
        del calibration_backend
    engine = ShadowKVEngine(
        backend=backend,
        max_memory_mb=args.max_memory_mb,
        speculative_k=args.speculative_k,
        idle_threshold_ms=args.idle_threshold_ms,
        policy=CostAwareSlackPolicy(**shadowkv_policy_kwargs),
        enable_gpu_tier=args.device.startswith('cuda'),
    )
    return backend, engine, shadowkv_policy_kwargs, shadowkv_policy_calibration


def _build_native_prefix_cache(args, backend_name: str):
    backend = load_backend(
        backend_name,
        model_name=benchmark.resolve_model(args.model),
        device=args.device,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        trust_remote_code=args.trust_remote_code,
    )
    engine = NativePrefixCachingEngine(backend=backend, max_memory_mb=args.max_memory_mb)
    return backend, engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='auto')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--workload', choices=['synthetic', 'public_dataset'], default='public_dataset')
    parser.add_argument('--variant', choices=sorted(SYNTHETIC_VARIANTS.keys()), default='high_skew')
    parser.add_argument('--dataset', choices=list_datasets(), default='daily_dialog')
    parser.add_argument('--prompt_mode', choices=['auto', *list_prompt_modes()], default='auto')
    parser.add_argument('--dataset_split', default=None)
    parser.add_argument('--n_requests', type=int, default=64)
    parser.add_argument('--simulate_arrivals', dest='simulate_arrivals', action='store_true')
    parser.add_argument('--disable_arrival_simulation', dest='simulate_arrivals', action='store_false')
    parser.set_defaults(simulate_arrivals=True)
    parser.add_argument('--mean_inter_arrival_ms', type=float, default=None)
    parser.add_argument('--max_arrival_sleep_ms', type=float, default=500.0)
    parser.add_argument('--max_memory_mb', type=int, default=64)
    parser.add_argument('--speculative_k', type=int, default=2)
    parser.add_argument('--idle_threshold_ms', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results_external_baselines')
    args = parser.parse_args()

    args.resolved_prompt_mode = benchmark.resolve_prompt_mode(args)
    set_seed(args.seed)
    requests = benchmark.build_requests(args)

    summary: dict[str, object] = {}
    capabilities: dict[str, object] = {}

    shadow_policy_kwargs = None
    shadow_policy_calibration = None

    set_seed(args.seed)
    summary['vllm_prefix_cache'], capabilities['vllm_prefix_cache'] = _run_baseline(
        'vllm_prefix_cache',
        lambda: _build_native_prefix_cache(args, 'vllm'),
        requests,
        args.simulate_arrivals,
        args.max_arrival_sleep_ms,
    )

    set_seed(args.seed)
    summary['sglang_prefix_cache'], capabilities['sglang_prefix_cache'] = _run_baseline(
        'sglang_prefix_cache',
        lambda: _build_native_prefix_cache(args, 'sglang'),
        requests,
        args.simulate_arrivals,
        args.max_arrival_sleep_ms,
    )

    set_seed(args.seed)
    try:
        shadow_backend, shadow_engine, shadow_policy_kwargs, shadow_policy_calibration = _build_shadow_kv(args)
        capabilities['shadow_kv'] = {
            'supports_external_kv': getattr(shadow_backend, 'supports_external_kv', False),
            'supports_native_prefix_caching': getattr(shadow_backend, 'supports_native_prefix_caching', False),
            'backend_name': getattr(shadow_backend, 'backend_name', 'unknown'),
        }
        summary['shadow_kv'] = _run_engine(
            'shadow_kv',
            shadow_backend,
            shadow_engine,
            requests,
            args.simulate_arrivals,
            args.max_arrival_sleep_ms,
        )
    except Exception as exc:
        _release_cuda_memory()
        summary['shadow_kv'] = {
            'backend': 'shadow_kv',
            'engine': 'shadow_kv',
            'error': f'{exc.__class__.__name__}: {exc}',
            'status': 'failed',
        }
        capabilities['shadow_kv'] = {
            'supports_external_kv': None,
            'supports_native_prefix_caching': None,
            'backend_name': 'shadow_kv',
        }

    pairwise = {}
    if 'mean_latency_ms' in summary.get('shadow_kv', {}) and 'mean_latency_ms' in summary.get('vllm_prefix_cache', {}):
        pairwise['shadow_kv_vs_vllm_mean_latency_ratio'] = float(summary['vllm_prefix_cache']['mean_latency_ms']) / max(float(summary['shadow_kv']['mean_latency_ms']), 1e-9)
    if 'mean_latency_ms' in summary.get('shadow_kv', {}) and 'mean_latency_ms' in summary.get('sglang_prefix_cache', {}):
        pairwise['shadow_kv_vs_sglang_mean_latency_ratio'] = float(summary['sglang_prefix_cache']['mean_latency_ms']) / max(float(summary['shadow_kv']['mean_latency_ms']), 1e-9)
    if 'mean_latency_ms' in summary.get('vllm_prefix_cache', {}) and 'mean_latency_ms' in summary.get('sglang_prefix_cache', {}):
        pairwise['vllm_vs_sglang_mean_latency_ratio'] = float(summary['sglang_prefix_cache']['mean_latency_ms']) / max(float(summary['vllm_prefix_cache']['mean_latency_ms']), 1e-9)
    summary['pairwise'] = pairwise
    summary['config'] = vars(args)
    summary['config']['resolved_model'] = benchmark.resolve_model(args.model)
    summary['config']['shadowkv_policy_kwargs'] = shadow_policy_kwargs
    summary['config']['shadowkv_policy_calibration'] = shadow_policy_calibration
    summary['capabilities'] = capabilities

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = (benchmark.resolve_model(args.model) or args.model).replace('/', '_').replace(':', '_')
    workload_slug = (
        f'{args.dataset}_{args.resolved_prompt_mode}'
        if args.workload == 'public_dataset'
        else args.variant
    )
    name = f'external_baselines_{model_slug}_{args.workload}_{workload_slug}_{args.device.replace(":", "_")}.json'
    out_file = output_dir / name
    out_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved to {out_file}')


if __name__ == '__main__':
    main()
