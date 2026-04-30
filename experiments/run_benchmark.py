from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import statistics
import time

from proactive_kv_cache.datasets import list_datasets, list_prompt_modes
from proactive_kv_cache.engines import (
    AdmissionControlledRuntimeCacheEngine,
    FrequencySpeculativeEngine,
    GreedyPrefixCacheEngine,
    NativePrefixCachingEngine,
    NoCacheEngine,
    ReactivePrefixCacheEngine,
    RuntimeNativeCacheEngine,
    ShadowKVEngine,
    ShadowKVPlusEngine,
    StrictReactivePrefixCacheEngine,
    maybe_shutdown,
    summarize_engine,
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


ALL_ENGINE_NAMES = [
    'no_cache',
    'native_prefix_cache',
    'reactive_prefix_cache',
    'greedy_prefix_cache',
    'strict_reactive_prefix_cache',
    'frequency_speculative',
    'shadow_kv',
    'shadow_kv_plus',
    'shadow_kv_plus_best_latency',
    'shadow_kv_plus_raw_observer',
    'shadow_kv_plus_scaffold_only',
    'shadow_kv_plus_early_layer',
    'shadow_kv_plus_logit_guard',
    'vllm_apc',
    'vllm_apc_shadowkv_plus',
    'sglang_radix_attention',
    'sglang_radix_attention_shadowkv_plus',
    'lmcache',
    'lmcache_shadowkv_plus',
]


RUNTIME_BASELINE_NAMES = [
    'vllm_apc',
    'vllm_apc_shadowkv_plus',
]

EXTERNAL_RUNTIME_BASELINE_NAMES = [
    'sglang_radix_attention',
    'sglang_radix_attention_shadowkv_plus',
    'lmcache',
    'lmcache_shadowkv_plus',
]


RUNTIME_BASELINE_FAMILIES = {
    'vllm_apc': 'vllm_apc',
    'vllm_apc_shadowkv_plus': 'vllm_apc',
}


def resolve_model(model: str | None) -> str | None:
    if model is None:
        return None
    return MODEL_PRESETS.get(model, model)


def build_requests(args):
    resolved_prompt_mode = getattr(args, 'resolved_prompt_mode', 'raw')
    if args.workload == 'synthetic':
        return make_synthetic_workload(
            variant=args.variant,
            n_requests=args.n_requests,
            seed=args.seed,
            mean_inter_arrival_ms=args.mean_inter_arrival_ms,
        )
    if args.workload == 'public_dataset':
        return make_public_dataset_workload(
            dataset_name=args.dataset,
            split=args.dataset_split,
            n_requests=args.n_requests,
            seed=args.seed,
            mean_inter_arrival_ms=args.mean_inter_arrival_ms or 150.0,
            prompt_mode=resolved_prompt_mode,
        )
    raise ValueError(f'Unsupported workload: {args.workload}')


def resolve_prompt_mode(args) -> str:
    if args.prompt_mode != 'auto':
        return args.prompt_mode
    if args.workload == 'public_dataset':
        return 'templated'
    return 'raw'


def maybe_sleep(current_idx: int, requests, simulate_arrivals: bool, max_sleep_ms: float) -> None:
    if not simulate_arrivals or current_idx == 0:
        return
    delay = requests[current_idx].arrival_time - requests[current_idx - 1].arrival_time
    if delay > 0:
        time.sleep(min(delay, max_sleep_ms / 1000.0))


def _profile_shadowkv_costs(backend) -> dict:
    max_len = 96 if backend.device.startswith('cuda') else 64
    candidate_lengths = [length for length in (8, 16, 32, 48, 64, 96) if length <= max_len]
    prompt = ('profiling token ' * (max_len + 16)).strip()
    tokens = backend.tokenize(prompt)
    if len(tokens) < max_len:
        prompt = ('profiling token ' * (max_len * 4)).strip()
        tokens = backend.tokenize(prompt)

    latencies = []
    lengths = []
    if candidate_lengths:
        warmup_tokens = tuple(tokens[: candidate_lengths[-1]])
        if warmup_tokens:
            try:
                backend.prefill(warmup_tokens)
            except Exception as exc:
                raise RuntimeError('ShadowKV calibration warmup failed before benchmarking') from exc
    for length in candidate_lengths:
        sample = tuple(tokens[:length])
        if len(sample) < length:
            continue
        timings = []
        repeats = 4 if backend.device.startswith('cuda') else 2
        for _ in range(repeats):
            timings.append(float(backend.prefill(sample).latency_ms))
        latencies.append(statistics.median(timings))
        lengths.append(length)

    if not lengths:
        estimate = max(float(backend.estimate_prefill_cost_ms(32)) / 32.0, 0.1)
        kv_mb_per_token = max(float(backend.estimate_kv_cache_bytes(32)) / (32.0 * 1024.0 * 1024.0), 0.0005)
        return {
            'profile_lengths': [],
            'profile_latencies_ms': [],
            'profile_ms_per_token': [],
            'token_benefit_ms': estimate,
            'speculation_penalty_ms': max(estimate * 4.0, 2.0),
            'ms_per_token': estimate,
            'fixed_prefill_overhead_ms': max(estimate * 4.0, 2.0),
            'kv_mb_per_token': kv_mb_per_token,
        }

    measured_ms_per_token = [lat / max(length, 1) for length, lat in zip(lengths, latencies)]
    estimate_slope = (
        float(backend.estimate_prefill_cost_ms(lengths[-1]) - backend.estimate_prefill_cost_ms(lengths[0]))
        / max(lengths[-1] - lengths[0], 1)
    )
    median_measured = statistics.median(measured_ms_per_token)
    if backend.device.startswith('cuda'):
        token_benefit_ms = max(estimate_slope, median_measured * 0.65, 0.20)
        speculation_penalty_ms = min(max(min(latencies) * 0.20, 2.0), 12.0)
    else:
        token_benefit_ms = max(estimate_slope, median_measured * 0.70, 0.35)
        speculation_penalty_ms = min(max(min(latencies) * 0.25, 4.0), 20.0)
    return {
        'profile_lengths': lengths,
        'profile_latencies_ms': latencies,
        'profile_ms_per_token': measured_ms_per_token,
        'token_benefit_ms': float(token_benefit_ms),
        'speculation_penalty_ms': float(speculation_penalty_ms),
        'ms_per_token': float(token_benefit_ms),
        'fixed_prefill_overhead_ms': float(speculation_penalty_ms),
        'kv_mb_per_token': max(float(backend.estimate_kv_cache_bytes(lengths[-1])) / (max(lengths[-1], 1) * 1024.0 * 1024.0), 0.0005),
    }


def _build_shadowkv_policy_kwargs(backend, prompt_mode: str = 'raw') -> tuple[dict, dict]:
    calibration = _profile_shadowkv_costs(backend)
    token_benefit_ms = calibration['ms_per_token']
    speculation_penalty_ms = calibration['fixed_prefill_overhead_ms']
    kv_mb_per_token = calibration['kv_mb_per_token']
    scaffold_mode = prompt_mode in ('templated', 'rag', 'semantic')
    if backend.device.startswith('cuda'):
        policy_kwargs = {
            'min_frequency': 0.16,
            'ms_per_token': token_benefit_ms,
            'fixed_prefill_overhead_ms': speculation_penalty_ms,
            'memory_penalty_per_mb': 0.75,
            'kv_mb_per_token': kv_mb_per_token,
            'idle_cost_fraction': 0.55,
            'gpu_idle_cost_fraction': 0.30,
            'benefit_cost_ratio': 0.92,
            'max_admissions_per_idle': 2,
            'min_prefix_len': 16,
            'preferred_prefix_len': 64,
            'max_prefix_len': 192,
            'min_observations': 3,
            'min_expected_net_ms': max(1.0, token_benefit_ms * 6.0),
            'min_recent_support': 0.04,
            'recent_support_weight': 0.60,
            'recent_streak_weight': 0.22,
            'observation_weight': 0.18,
            'global_frequency_weight': 0.22,
            'reuse_discount': 0.95,
            'min_utility_score': 0.02,
            'reuse_horizon_s': 1.35 if prompt_mode == 'rag' else (1.20 if prompt_mode == 'semantic' else (1.05 if scaffold_mode else 0.75)),
            'bootstrap_horizon_requests': 0.90 if scaffold_mode else 0.45,
            'branching_weight': 0.14 if scaffold_mode else 0.10,
        }
        if scaffold_mode:
            policy_kwargs['min_frequency'] = 0.08
            policy_kwargs['min_observations'] = 1
            policy_kwargs['min_recent_support'] = 0.01
            policy_kwargs['min_expected_net_ms'] = max(0.25, token_benefit_ms * 2.5)
            policy_kwargs['benefit_cost_ratio'] = 0.85
            policy_kwargs['reuse_horizon_s'] = 1.60 if prompt_mode == 'rag' else (1.50 if prompt_mode == 'semantic' else 1.40)
            policy_kwargs['bootstrap_horizon_requests'] = 1.25 if prompt_mode == 'rag' else (1.20 if prompt_mode == 'semantic' else 1.10)
            policy_kwargs['preferred_prefix_len'] = 80
            policy_kwargs['max_prefix_len'] = 224
            policy_kwargs['max_admissions_per_idle'] = 4
        return policy_kwargs, calibration
    policy_kwargs = {
        'min_frequency': 0.24,
        'ms_per_token': max(token_benefit_ms * 0.95, 0.2),
        'fixed_prefill_overhead_ms': max(speculation_penalty_ms * 1.1, 2.0),
        'memory_penalty_per_mb': 0.90,
        'kv_mb_per_token': kv_mb_per_token,
        'idle_cost_fraction': 0.75,
        'gpu_idle_cost_fraction': 0.75,
        'benefit_cost_ratio': 1.35,
        'max_admissions_per_idle': 2,
        'min_prefix_len': 12,
        'preferred_prefix_len': 48,
        'max_prefix_len': 128,
        'min_observations': 4,
        'min_expected_net_ms': max(6.0, token_benefit_ms * 8.0),
        'min_recent_support': 0.04,
        'recent_support_weight': 0.58,
        'recent_streak_weight': 0.20,
        'observation_weight': 0.18,
        'global_frequency_weight': 0.24,
        'reuse_discount': 0.92,
        'min_utility_score': 0.04,
        'reuse_horizon_s': 1.10 if prompt_mode == 'rag' else (1.00 if prompt_mode == 'semantic' else (0.90 if scaffold_mode else 0.60)),
        'bootstrap_horizon_requests': 0.80 if scaffold_mode else 0.35,
        'branching_weight': 0.14 if scaffold_mode else 0.10,
    }
    if scaffold_mode:
        policy_kwargs['min_frequency'] = 0.14
        policy_kwargs['min_observations'] = 3
        policy_kwargs['min_recent_support'] = 0.01
        policy_kwargs['min_expected_net_ms'] = max(2.0, token_benefit_ms * 4.0)
        policy_kwargs['benefit_cost_ratio'] = 1.15
        policy_kwargs['reuse_horizon_s'] = 1.25 if prompt_mode == 'rag' else 1.05
        policy_kwargs['bootstrap_horizon_requests'] = 0.95 if prompt_mode == 'rag' else 0.85
    return policy_kwargs, calibration


def load_backend_from_args(args):
    return load_backend(
        args.backend,
        model_name=resolve_model(args.model),
        device=args.device,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=not args.disable_native_prefix_caching,
        trust_remote_code=getattr(args, 'trust_remote_code', False),
    )


def list_engine_names(args) -> list[str]:
    if getattr(args, 'engines', None):
        return list(args.engines)
    names = ['no_cache']
    if args.backend == 'vllm':
        names.append('native_prefix_cache')
        if getattr(args, 'include_runtime_baselines', False):
            names.extend(RUNTIME_BASELINE_NAMES)
        return names
    names.extend(
        [
            'reactive_prefix_cache',
            'greedy_prefix_cache',
            'strict_reactive_prefix_cache',
        ]
    )
    if args.include_experimental:
        names.extend(['frequency_speculative', 'shadow_kv', 'shadow_kv_plus', 'shadow_kv_plus_best_latency', 'shadow_kv_plus_raw_observer'])
    if getattr(args, 'include_semantic_ablations', False):
        names.extend([
            'shadow_kv_plus_scaffold_only',
            'shadow_kv_plus_early_layer',
            'shadow_kv_plus_logit_guard',
        ])
    if getattr(args, 'include_runtime_baselines', False) and args.backend == 'vllm':
        names.extend(RUNTIME_BASELINE_NAMES)
    return names


def build_engine(args, backend, engine_name: str):
    if engine_name == 'no_cache':
        return NoCacheEngine(backend=backend, max_memory_mb=args.max_memory_mb)
    if engine_name == 'native_prefix_cache':
        return NativePrefixCachingEngine(backend=backend, max_memory_mb=args.max_memory_mb)
    if engine_name in EXTERNAL_RUNTIME_BASELINE_NAMES:
        raise ValueError(
            f'{engine_name} is an external runtime baseline. Run it through '
            'literature_accurate_baselines/run_runtime_cache_baseline.py so the measured system is the actual runtime.'
        )
    if engine_name in RUNTIME_BASELINE_NAMES:
        if args.backend != 'vllm':
            raise ValueError(f'{engine_name} requires --backend vllm for literature-accurate vLLM APC measurement.')
        runtime_family = RUNTIME_BASELINE_FAMILIES[engine_name]
        if engine_name.endswith('_shadowkv_plus'):
            return AdmissionControlledRuntimeCacheEngine(
                backend=backend,
                max_memory_mb=args.max_memory_mb,
                name=engine_name,
                runtime_family=runtime_family,
            )
        return RuntimeNativeCacheEngine(
            backend=backend,
            max_memory_mb=args.max_memory_mb,
            name=engine_name,
            runtime_family=runtime_family,
        )
    if engine_name == 'reactive_prefix_cache':
        return ReactivePrefixCacheEngine(backend=backend, max_memory_mb=args.max_memory_mb)
    if engine_name == 'greedy_prefix_cache':
        return GreedyPrefixCacheEngine(backend=backend, max_memory_mb=args.max_memory_mb)
    if engine_name == 'strict_reactive_prefix_cache':
        return StrictReactivePrefixCacheEngine(backend=backend, max_memory_mb=args.max_memory_mb)
    if engine_name == 'frequency_speculative':
        return FrequencySpeculativeEngine(
            backend=backend,
            max_memory_mb=args.max_memory_mb,
            speculative_k=args.speculative_k,
            idle_threshold_ms=args.idle_threshold_ms,
        )
    if engine_name in ('shadow_kv', 'shadow_kv_plus', 'shadow_kv_plus_best_latency', 'shadow_kv_plus_raw_observer', 'shadow_kv_plus_scaffold_only', 'shadow_kv_plus_early_layer', 'shadow_kv_plus_logit_guard'):
        calibration_backend = load_backend_from_args(args)
        try:
            shadowkv_policy_kwargs, shadowkv_policy_calibration = _build_shadowkv_policy_kwargs(calibration_backend, prompt_mode=args.resolved_prompt_mode)
        finally:
            del calibration_backend
        args.shadowkv_policy_calibration = shadowkv_policy_calibration
        args.shadowkv_policy_kwargs = shadowkv_policy_kwargs
        if engine_name == 'shadow_kv':
            return ShadowKVEngine(
                backend=backend,
                max_memory_mb=args.max_memory_mb,
                speculative_k=args.speculative_k,
                idle_threshold_ms=args.idle_threshold_ms,
                policy=CostAwareSlackPolicy(**shadowkv_policy_kwargs),
                enable_gpu_tier=args.device.startswith('cuda'),
            )
        ablation_mode = {
            'shadow_kv_plus': 'safe',
            'shadow_kv_plus_best_latency': 'best_latency',
            'shadow_kv_plus_raw_observer': 'raw_observer',
            'shadow_kv_plus_scaffold_only': 'scaffold_only',
            'shadow_kv_plus_early_layer': 'early_layer',
            'shadow_kv_plus_logit_guard': 'logit_guard',
        }[engine_name]
        raw_strategy = {
            'shadow_kv_plus': 'strict_utility_gate',
            'shadow_kv_plus_best_latency': 'best_latency',
            'shadow_kv_plus_raw_observer': 'raw_observer',
        }.get(engine_name)
        return ShadowKVPlusEngine(
            backend=backend,
            max_memory_mb=args.max_memory_mb,
            speculative_k=args.speculative_k,
            idle_threshold_ms=args.idle_threshold_ms,
            policy=CostAwareSlackPolicy(**shadowkv_policy_kwargs),
            enable_gpu_tier=args.device.startswith('cuda'),
            semantic_ablation_mode=ablation_mode,
            raw_strategy=raw_strategy,
            early_layer_reuse_ratio=args.early_layer_reuse_ratio,
            logit_guard_threshold=args.logit_guard_threshold,
            allow_approximate_semantic_reuse=args.allow_unsafe_semantic_kv_reuse,
        )
    raise ValueError(f'Unknown engine name: {engine_name}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['fake', 'hf', 'vllm'], default='fake')
    parser.add_argument('--model', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='auto')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--disable_native_prefix_caching', action='store_true')
    parser.add_argument('--include_experimental', action='store_true')
    parser.add_argument('--engines', nargs='+', choices=ALL_ENGINE_NAMES, help='Run only the specified engines, in the given order.')
    parser.add_argument('--include_runtime_baselines', action='store_true', help='Add in-process literature-accurate vLLM APC variants when --backend vllm. Use literature_accurate_baselines/run_runtime_cache_baseline.py for SGLang and LMCache.')
    parser.add_argument('--include_semantic_ablations', action='store_true', help='Add scaffold-only, early-layer, and logit-guarded ShadowKV++ ablation engines.')
    parser.add_argument('--enable_policy_trace', action='store_true', help='Write per-request policy_trace.jsonl. Disabled by default for performance benchmarking.')
    parser.add_argument('--allow_unsafe_semantic_kv_reuse', action='store_true', help='Allow unguarded approximate semantic KV reuse. Intended only for fake/controlled ablations.')
    parser.add_argument('--early_layer_reuse_ratio', type=float, default=0.35, help='Fraction of semantic KV prefix reused in early-layer ablation.')
    parser.add_argument('--logit_guard_threshold', type=float, default=0.08, help='Maximum TV distance for logit-guard semantic reuse.')
    parser.add_argument('--workload', choices=['synthetic', 'public_dataset'], default='synthetic')
    parser.add_argument('--variant', choices=sorted(SYNTHETIC_VARIANTS.keys()), default='high_skew')
    parser.add_argument('--dataset', choices=list_datasets(), default='daily_dialog')
    parser.add_argument('--prompt_mode', choices=['auto', *list_prompt_modes()], default='auto')
    parser.add_argument('--dataset_split', default=None)
    parser.add_argument('--n_requests', type=int, default=60)
    parser.add_argument('--simulate_arrivals', dest='simulate_arrivals', action='store_true')
    parser.add_argument('--disable_arrival_simulation', dest='simulate_arrivals', action='store_false')
    parser.set_defaults(simulate_arrivals=True)
    parser.add_argument('--mean_inter_arrival_ms', type=float, default=None)
    parser.add_argument('--max_arrival_sleep_ms', type=float, default=500.0)
    parser.add_argument('--max_memory_mb', type=int, default=64)
    parser.add_argument('--speculative_k', type=int, default=2)
    parser.add_argument('--idle_threshold_ms', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()
    args.resolved_prompt_mode = resolve_prompt_mode(args)
    args.shadowkv_policy_calibration = None
    args.shadowkv_policy_kwargs = None

    set_seed(args.seed)
    requests = build_requests(args)
    engine_names = list_engine_names(args)
    summary = {}
    capabilities = None
    policy_trace_rows = []

    for engine_name in engine_names:
        set_seed(args.seed)
        backend = load_backend_from_args(args)
        if capabilities is None:
            capabilities = {
                'supports_external_kv': getattr(backend, 'supports_external_kv', False),
                'supports_native_prefix_caching': getattr(backend, 'supports_native_prefix_caching', False),
            }
        engine = build_engine(args, backend, engine_name)
        engine.trace_enabled = bool(args.enable_policy_trace)
        shared_prefix_token_cache = {}
        for idx, req in enumerate(requests):
            maybe_sleep(idx, requests, args.simulate_arrivals, args.max_arrival_sleep_ms)
            tokens = backend.tokenize(req.prompt)
            metadata = dict(req.metadata or {})
            metadata['arrival_time'] = req.arrival_time
            shared_prefix_text = metadata.get('shared_prefix_text')
            if shared_prefix_text:
                hint_len = shared_prefix_token_cache.get(shared_prefix_text)
                if hint_len is None:
                    hint_len = len(backend.tokenize(shared_prefix_text))
                    shared_prefix_token_cache[shared_prefix_text] = hint_len
                metadata['shared_prefix_hint_tokens'] = min(int(hint_len), len(tokens))
            engine.serve_tokens(req.request_id, tokens, metadata=metadata)
        maybe_shutdown(engine)
        summary[engine.name] = summarize_engine(engine)
        if args.enable_policy_trace:
            for row in getattr(engine, 'policy_trace_rows', []):
                traced = dict(row)
                traced.setdefault('engine', engine.name)
                traced['model'] = resolve_model(args.model)
                traced['backend'] = args.backend
                traced['device'] = args.device
                traced['dtype'] = args.dtype
                traced['workload'] = args.workload
                traced['dataset'] = args.dataset if args.workload == 'public_dataset' else None
                traced['variant'] = args.variant if args.workload == 'synthetic' else None
                traced['prompt_mode'] = args.resolved_prompt_mode
                traced['seed'] = args.seed
                policy_trace_rows.append(traced)
        del engine
        del backend

    if 'no_cache' in summary:
        baseline = summary['no_cache']
        for engine_summary in summary.values():
            engine_summary['speedup_vs_no_cache_mean'] = baseline['mean_latency_ms'] / max(engine_summary['mean_latency_ms'], 1e-9)
            engine_summary['speedup_vs_no_cache_p95'] = baseline['p95_latency_ms'] / max(engine_summary['p95_latency_ms'], 1e-9)
    summary['config'] = vars(args)
    summary['config']['resolved_model'] = resolve_model(args.model)
    summary['capabilities'] = capabilities or {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = (resolve_model(args.model) or args.model or 'default').replace('/', '_').replace(':', '_')
    workload_slug = (
        f"{args.dataset}_{args.resolved_prompt_mode}"
        if args.workload == 'public_dataset'
        else args.variant
    )
    name = f"benchmark_{args.backend}_{model_slug}_{args.workload}_{workload_slug}_{args.device.replace(':', '_')}.json"
    out_file = output_dir / name
    out_file.write_text(json.dumps(summary, indent=2))
    if args.enable_policy_trace:
        trace_file = output_dir / 'policy_trace.jsonl'
        with trace_file.open('w', encoding='utf-8') as fh:
            for row in policy_trace_rows:
                fh.write(json.dumps(row, sort_keys=True) + '\n')
        summary['config']['policy_trace_file'] = str(trace_file)
        summary['config']['policy_trace_rows'] = len(policy_trace_rows)
        out_file.write_text(json.dumps(summary, indent=2))
    else:
        summary['config']['policy_trace_file'] = None
        summary['config']['policy_trace_rows'] = 0
        out_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved to {out_file}')
    if args.enable_policy_trace:
        print(f'Saved policy trace to {trace_file} ({len(policy_trace_rows)} rows)')
    else:
        print('Policy trace disabled. Re-run with --enable_policy_trace to emit policy_trace.jsonl.')


if __name__ == '__main__':
    main()
