from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import csv
import json
import statistics
import time

from proactive_kv_cache.datasets import list_datasets, list_prompt_modes
from proactive_kv_cache.energy import NvidiaEnergyMeter, measure_idle_baseline
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
    ShadowKVPlusLiteEngine,
    StrictReactivePrefixCacheEngine,
    maybe_shutdown,
    summarize_engine,
)
from proactive_kv_cache.models import load_backend
from proactive_kv_cache.policy import CostAwareSlackPolicy
from proactive_kv_cache.config_loader import CONFIG
from proactive_kv_cache.telemetry import JsonlLogger, build_run_manifest
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
    'shadow_kv_plus_lite',
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

POLICY_PRESETS = [
    'conservative',
    'balanced',
    'aggressive_prefix',
    'aggressive_gpu',
    'low_latency',
]

DEFAULT_POLICY_TUNING_PRESETS = [
    'balanced',
    'aggressive_prefix',
    'aggressive_gpu',
]

SHADOWKV_POLICY_ENGINE_NAMES = {
    'shadow_kv',
    'shadow_kv_plus',
    'shadow_kv_plus_lite',
    'shadow_kv_plus_best_latency',
    'shadow_kv_plus_raw_observer',
    'shadow_kv_plus_scaffold_only',
    'shadow_kv_plus_early_layer',
    'shadow_kv_plus_logit_guard',
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


def prepare_request_metadata(backend, req, tokens, shared_prefix_token_cache: dict[str, int]) -> dict:
    metadata = dict(req.metadata or {})
    metadata['arrival_time'] = req.arrival_time
    shared_prefix_text = metadata.get('shared_prefix_text')
    if shared_prefix_text:
        hint_len = shared_prefix_token_cache.get(shared_prefix_text)
        if hint_len is None:
            hint_len = len(backend.tokenize(shared_prefix_text))
            shared_prefix_token_cache[shared_prefix_text] = hint_len
        metadata['shared_prefix_hint_tokens'] = min(int(hint_len), len(tokens))
    return metadata


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
            'prefill_ms_per_token': estimate,
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
        'prefill_ms_per_token': float(token_benefit_ms),
        'fixed_prefill_overhead_ms': float(speculation_penalty_ms),
        'kv_mb_per_token': max(float(backend.estimate_kv_cache_bytes(lengths[-1])) / (max(lengths[-1], 1) * 1024.0 * 1024.0), 0.0005),
    }


def _apply_shadowkv_policy_preset(
    policy_kwargs: dict,
    calibration: dict,
    *,
    preset: str,
    prompt_mode: str,
    is_cuda: bool,
) -> dict:
    preset = str(preset or 'balanced')
    if preset not in POLICY_PRESETS:
        raise ValueError(f'Unknown policy preset: {preset}')
    tuned = dict(policy_kwargs)
    token_benefit_ms = float(calibration.get('prefill_ms_per_token', tuned.get('prefill_ms_per_token', 1.0)))
    scaffold_mode = prompt_mode in ('templated', 'rag', 'semantic')

    if preset == 'balanced':
        return tuned
    if preset == 'conservative':
        tuned['min_frequency'] = max(float(tuned['min_frequency']) * 1.6, 0.20 if scaffold_mode else 0.30)
        tuned['min_observations'] = max(int(tuned['min_observations']) + 2, 3)
        tuned['min_recent_support'] = max(float(tuned['min_recent_support']) * 1.5, 0.04)
        tuned['min_expected_net_ms'] = max(float(tuned['min_expected_net_ms']) * 1.6, token_benefit_ms * 6.0)
        tuned['benefit_cost_ratio'] = max(float(tuned['benefit_cost_ratio']), 1.20)
        tuned['max_admissions_per_idle'] = 1
        tuned['reuse_horizon_s'] = max(float(tuned['reuse_horizon_s']) * 0.75, 0.45)
        tuned['bootstrap_horizon_requests'] = max(float(tuned['bootstrap_horizon_requests']) * 0.65, 0.25)
        return tuned
    if preset == 'aggressive_prefix':
        tuned['min_frequency'] = min(float(tuned['min_frequency']), 0.04 if scaffold_mode else 0.10)
        tuned['min_observations'] = 1 if scaffold_mode else 2
        tuned['min_recent_support'] = min(float(tuned['min_recent_support']), 0.005 if scaffold_mode else 0.02)
        tuned['min_expected_net_ms'] = max(0.10, token_benefit_ms * (1.25 if scaffold_mode else 2.50))
        tuned['benefit_cost_ratio'] = min(float(tuned['benefit_cost_ratio']), 0.72 if scaffold_mode else 0.90)
        tuned['max_admissions_per_idle'] = max(int(tuned['max_admissions_per_idle']), 5 if scaffold_mode else 3)
        tuned['preferred_prefix_len'] = max(int(tuned['preferred_prefix_len']), 96 if scaffold_mode else 72)
        tuned['max_prefix_len'] = max(int(tuned['max_prefix_len']), 256 if scaffold_mode else 192)
        tuned['reuse_horizon_s'] = max(float(tuned['reuse_horizon_s']), 1.90 if scaffold_mode else 1.10)
        tuned['bootstrap_horizon_requests'] = max(float(tuned['bootstrap_horizon_requests']), 1.45 if scaffold_mode else 0.75)
        tuned['branching_weight'] = max(float(tuned['branching_weight']), 0.18 if scaffold_mode else 0.12)
        return tuned
    if preset == 'aggressive_gpu':
        tuned = _apply_shadowkv_policy_preset(
            tuned,
            calibration,
            preset='aggressive_prefix',
            prompt_mode=prompt_mode,
            is_cuda=is_cuda,
        )
        if is_cuda:
            tuned['gpu_idle_cost_fraction'] = min(float(tuned['gpu_idle_cost_fraction']), 0.12)
            tuned['idle_cost_fraction'] = min(float(tuned['idle_cost_fraction']), 0.35)
            tuned['memory_penalty_per_mb'] = min(float(tuned['memory_penalty_per_mb']), 0.30)
            tuned['min_expected_net_ms'] = max(0.05, token_benefit_ms * (0.90 if scaffold_mode else 1.75))
            tuned['benefit_cost_ratio'] = min(float(tuned['benefit_cost_ratio']), 0.65 if scaffold_mode else 0.82)
        return tuned
    if preset == 'low_latency':
        tuned['min_frequency'] = min(float(tuned['min_frequency']), 0.06 if scaffold_mode else 0.12)
        tuned['min_observations'] = 1
        tuned['min_expected_net_ms'] = max(0.05, token_benefit_ms * 1.0)
        tuned['benefit_cost_ratio'] = min(float(tuned['benefit_cost_ratio']), 0.70)
        tuned['max_admissions_per_idle'] = max(int(tuned['max_admissions_per_idle']), 4)
        tuned['gpu_idle_cost_fraction'] = min(float(tuned['gpu_idle_cost_fraction']), 0.16 if is_cuda else 0.50)
        tuned['reuse_horizon_s'] = max(float(tuned['reuse_horizon_s']), 1.60 if scaffold_mode else 1.00)
        tuned['bootstrap_horizon_requests'] = max(float(tuned['bootstrap_horizon_requests']), 1.20 if scaffold_mode else 0.60)
        return tuned
    return tuned


def _build_shadowkv_policy_kwargs_from_calibration(
    calibration: dict,
    *,
    device: str,
    prompt_mode: str = 'raw',
    preset: str = 'balanced',
) -> tuple[dict, dict]:
    token_benefit_ms = calibration['prefill_ms_per_token']
    speculation_penalty_ms = calibration['fixed_prefill_overhead_ms']
    kv_mb_per_token = calibration['kv_mb_per_token']
    scaffold_mode = prompt_mode in ('templated', 'rag', 'semantic')
    is_cuda = str(device).startswith('cuda')
    if is_cuda:
        policy_kwargs = {
            'min_frequency': 0.16,
            'prefill_ms_per_token': token_benefit_ms,
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
        policy_kwargs = _apply_shadowkv_policy_preset(policy_kwargs, calibration, preset=preset, prompt_mode=prompt_mode, is_cuda=is_cuda)
        calibration = dict(calibration)
        calibration['policy_preset'] = preset
        return policy_kwargs, calibration
    policy_kwargs = {
        'min_frequency': 0.24,
        'prefill_ms_per_token': max(token_benefit_ms * 0.95, 0.2),
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
    policy_kwargs = _apply_shadowkv_policy_preset(policy_kwargs, calibration, preset=preset, prompt_mode=prompt_mode, is_cuda=is_cuda)
    calibration = dict(calibration)
    calibration['policy_preset'] = preset
    return policy_kwargs, calibration


def _build_shadowkv_policy_kwargs(backend, prompt_mode: str = 'raw', preset: str = 'balanced') -> tuple[dict, dict]:
    calibration = _profile_shadowkv_costs(backend)
    return _build_shadowkv_policy_kwargs_from_calibration(
        calibration,
        device=backend.device,
        prompt_mode=prompt_mode,
        preset=preset,
    )


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
        names.extend(['frequency_speculative', 'shadow_kv', 'shadow_kv_plus', 'shadow_kv_plus_lite', 'shadow_kv_plus_best_latency', 'shadow_kv_plus_raw_observer'])
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
    if engine_name in SHADOWKV_POLICY_ENGINE_NAMES:
        preset = str(getattr(args, 'policy_preset', 'balanced') or 'balanced')
        base_calibration = getattr(args, '_shadowkv_policy_calibration_base', None)
        if base_calibration is None:
            calibration_backend = load_backend_from_args(args)
            try:
                base_calibration = _profile_shadowkv_costs(calibration_backend)
            finally:
                del calibration_backend
        shadowkv_policy_kwargs, shadowkv_policy_calibration = _build_shadowkv_policy_kwargs_from_calibration(
            base_calibration,
            device=args.device,
            prompt_mode=args.resolved_prompt_mode,
            preset=preset,
        )
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
        if engine_name == 'shadow_kv_plus_lite':
            return ShadowKVPlusLiteEngine(
                backend=backend,
                max_memory_mb=args.max_memory_mb,
                speculative_k=0,
                idle_threshold_ms=args.idle_threshold_ms,
                policy=CostAwareSlackPolicy(**shadowkv_policy_kwargs),
                enable_gpu_tier=args.device.startswith('cuda'),
                min_reuse_prefix_tokens=args.min_reuse_prefix_tokens,
                enable_utility_admission=not bool(args.disable_utility_admission),
                utility_min_net_saved_ms=args.utility_min_net_saved_ms,
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


def _warmup_backend(backend, requests) -> None:
    if not requests:
        return
    warmup_tokens = backend.tokenize(requests[0].prompt)[:256]
    for _ in range(3):
        try:
            backend.prefill(warmup_tokens)
        except Exception:
            pass
    if getattr(backend, 'device', '').startswith('cuda'):
        import torch

        torch.cuda.synchronize()


def _policy_tuning_score(metrics: dict, metric: str) -> float:
    if metric == 'mean_latency_ms':
        return float(metrics.get('mean_latency_ms', 0.0))
    if metric == 'p95_latency_ms':
        return float(metrics.get('p95_latency_ms', 0.0))
    if metric == 'utility_adjusted_latency':
        mean_latency = float(metrics.get('mean_latency_ms', 0.0))
        p95_latency = float(metrics.get('p95_latency_ms', mean_latency))
        wasted_ratio = float(metrics.get('wasted_compute_ratio', 0.0))
        utility = float(metrics.get('policy_net_utility_ms', metrics.get('utility_proxy_ms', 0.0)))
        return mean_latency + 0.10 * p95_latency + 20.0 * wasted_ratio - 0.01 * utility
    raise ValueError(f'Unsupported policy tuning metric: {metric}')


def _run_engine_requests(args, backend, engine, requests, *, simulate_arrivals: bool) -> None:
    shared_prefix_token_cache: dict[str, int] = {}
    for idx, req in enumerate(requests):
        maybe_sleep(idx, requests, simulate_arrivals, args.max_arrival_sleep_ms)
        tokens = backend.tokenize(req.prompt)
        metadata = prepare_request_metadata(backend, req, tokens, shared_prefix_token_cache)
        engine.serve_tokens(req.request_id, tokens, metadata=metadata)


def _should_run_policy_tuning(args, engine_names: list[str]) -> bool:
    if not getattr(args, 'enable_policy_tuning', False):
        return False
    if args.backend not in {'fake', 'hf'}:
        return False
    return any(name in SHADOWKV_POLICY_ENGINE_NAMES for name in engine_names)


def _run_shadowkv_policy_tuning(args, requests, output_dir: Path, engine_names: list[str]) -> dict | None:
    if not _should_run_policy_tuning(args, engine_names):
        return None
    tuning_requests = list(requests[: max(int(args.policy_tuning_requests), 0)])
    if not tuning_requests:
        return None

    candidates = []
    for preset in args.policy_tuning_presets:
        if preset not in POLICY_PRESETS:
            raise ValueError(f'Unknown policy tuning preset: {preset}')
        if preset not in candidates:
            candidates.append(preset)
    if not candidates:
        return None

    calibration_backend = load_backend_from_args(args)
    try:
        args._shadowkv_policy_calibration_base = _profile_shadowkv_costs(calibration_backend)
    finally:
        del calibration_backend

    original_preset = args.policy_preset
    rows = []
    selected = None
    best_score = None
    for offset, preset in enumerate(candidates):
        args.policy_preset = preset
        set_seed(args.seed + 1009 + offset)
        backend = load_backend_from_args(args)
        engine = build_engine(args, backend, 'shadow_kv_plus')
        engine.trace_enabled = False
        engine.decision_logger = None
        try:
            _warmup_backend(backend, tuning_requests)
            started = time.perf_counter()
            _run_engine_requests(args, backend, engine, tuning_requests, simulate_arrivals=False)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            maybe_shutdown(engine)
            metrics = summarize_engine(engine)
            score = _policy_tuning_score(metrics, args.policy_tuning_metric)
            rows.append({
                'preset': preset,
                'score': score,
                'elapsed_ms': elapsed_ms,
                'metrics': metrics,
                'policy_kwargs': dict(getattr(args, 'shadowkv_policy_kwargs', {}) or {}),
                'policy_calibration': dict(getattr(args, 'shadowkv_policy_calibration', {}) or {}),
            })
            if best_score is None or score < best_score:
                best_score = score
                selected = preset
        finally:
            del engine
            del backend

    args.policy_preset = selected or original_preset
    report = {
        'enabled': True,
        'method': 'fixed_preset_calibration',
        'selected_preset': args.policy_preset,
        'selection_metric': args.policy_tuning_metric,
        'tuning_requests': len(tuning_requests),
        'candidate_presets': candidates,
        'candidates': rows,
        'excluded_from_measured_run': True,
    }
    args.policy_tuning_report = report
    report_file = output_dir / 'policy_tuning_report.json'
    report_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['report_file'] = str(report_file)
    return report


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
    parser.add_argument('--policy_preset', choices=POLICY_PRESETS, default='balanced', help='Fixed ShadowKV/ShadowKV++ policy preset used for the measured run.')
    parser.add_argument('--enable_policy_tuning', action='store_true', help='Run a short unmeasured ShadowKV++ preset calibration before the measured run.')
    parser.add_argument('--policy_tuning_requests', type=int, default=24, help='Number of leading requests used for unmeasured preset calibration.')
    parser.add_argument('--policy_tuning_presets', nargs='+', choices=POLICY_PRESETS, default=DEFAULT_POLICY_TUNING_PRESETS, help='Candidate presets for --enable_policy_tuning.')
    parser.add_argument('--policy_tuning_metric', choices=['mean_latency_ms', 'p95_latency_ms', 'utility_adjusted_latency'], default='utility_adjusted_latency', help='Metric minimized during preset calibration.')
    parser.add_argument('--enable_policy_trace', action='store_true', help='Write per-request policy_trace.jsonl. Disabled by default for performance benchmarking.')
    parser.add_argument('--enable_decision_log', action='store_true', help='Write structured per-request policy decisions as JSONL.')
    parser.add_argument('--config_path', default=None, help='Path to the versioned ShadowKV policy config.')
    parser.add_argument('--semantic_index_diagnostics', action='store_true', help='Write semantic index diagnostic JSON for each semantic-capable engine.')
    parser.add_argument('--allow_unsafe_semantic_kv_reuse', action='store_true', help='Allow unguarded approximate semantic KV reuse. Intended only for fake/controlled ablations.')
    parser.add_argument('--early_layer_reuse_ratio', type=float, default=0.35, help='Fraction of semantic KV prefix reused in early-layer ablation.')
    parser.add_argument('--logit_guard_threshold', type=float, default=0.08, help='Maximum TV distance for logit-guard semantic reuse.')
    parser.add_argument('--min_reuse_prefix_tokens', type=int, default=None, help='Override ShadowKV++ Lite break-even reuse prefix threshold. Defaults are backend/device-aware.')
    parser.add_argument('--disable_utility_admission', action='store_true', help='Disable Phase 3 online utility-aware admission for ShadowKV++ Lite.')
    parser.add_argument('--utility_min_net_saved_ms', type=float, default=None, help='Minimum estimated net utility required for ShadowKV++ Lite reuse. Defaults to config policy.lite.utility_min_net_saved_ms.')
    parser.add_argument('--measure_energy', action='store_true', help='Measure GPU energy per engine using NVML total-energy deltas when available.')
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU index used by NVML/nvidia-smi energy measurement.')
    parser.add_argument('--idle_baseline_seconds', type=float, default=0.0, help='Optional idle-energy baseline duration measured before engine runs. Use 5-30 seconds for paper runs.')
    parser.add_argument('--energy_output_csv', default=None, help='Optional CSV file for per-engine latency and energy summary. Defaults beside JSON when --measure_energy is used.')
    parser.add_argument('--workload', choices=['synthetic', 'public_dataset'], default='synthetic')
    parser.add_argument('--variant', choices=sorted(SYNTHETIC_VARIANTS.keys()), default='high_skew')
    parser.add_argument('--dataset', choices=list_datasets(), default='daily_dialog')
    parser.add_argument('--prompt_mode', choices=['auto', *list_prompt_modes()], default='auto')
    parser.add_argument('--dataset_split', default=None)
    parser.add_argument('--n_requests', type=int, default=128)
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
    if args.config_path:
        CONFIG.load(args.config_path)
    args.resolved_prompt_mode = resolve_prompt_mode(args)
    args.shadowkv_policy_calibration = None
    args.shadowkv_policy_kwargs = None
    args.policy_tuning_report = None
    args._shadowkv_policy_calibration_base = None

    set_seed(args.seed)
    requests = build_requests(args)
    engine_names = list_engine_names(args)
    summary = {}
    capabilities = None
    policy_trace_rows = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_shadowkv_policy_tuning(args, requests, output_dir, engine_names)
    manifest = build_run_manifest(args=args, config_snapshot=CONFIG.snapshot(), config_hash=CONFIG.file_hash)
    manifest_file = output_dir / 'run_manifest.json'
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    decision_logger = None
    decision_log_enabled = bool(args.enable_decision_log or CONFIG.get('telemetry.enabled', False))
    if decision_log_enabled:
        decision_logger = JsonlLogger(output_dir / str(CONFIG.get('telemetry.decision_log_name', 'policy_decisions.jsonl')))

    energy_meter = NvidiaEnergyMeter(args.gpu_index) if args.measure_energy else None
    idle_energy_report = None
    idle_power_w = None
    if energy_meter is not None and float(args.idle_baseline_seconds or 0.0) > 0.0:
        idle_energy_report = measure_idle_baseline(energy_meter, float(args.idle_baseline_seconds))
        idle_power_w = idle_energy_report.get('idle_baseline_power_w')

    try:
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
            engine.decision_logger = decision_logger
            
            _warmup_backend(backend, requests)
            energy_before = energy_meter.snapshot() if energy_meter is not None else None
            _run_engine_requests(args, backend, engine, requests, simulate_arrivals=args.simulate_arrivals)
            energy_after = energy_meter.snapshot() if energy_meter is not None else None
            maybe_shutdown(engine)
            engine_summary = summarize_engine(engine)
            if energy_meter is not None and energy_before is not None and energy_after is not None:
                energy_metrics = NvidiaEnergyMeter.delta(energy_before, energy_after)
                gpu_energy_j = energy_metrics.get('gpu_energy_j')
                elapsed_s = float(energy_metrics.get('energy_elapsed_s') or 0.0)
                idle_adjusted_j = None
                if gpu_energy_j is not None and idle_power_w is not None:
                    idle_adjusted_j = max(float(gpu_energy_j) - float(idle_power_w) * elapsed_s, 0.0)
                n_req = max(len(requests), 1)
                total_recomputed = max(float(engine_summary.get('recompute_tokens_total', 0.0)), 0.0)
                total_reused = max(float(engine_summary.get('reused_prefix_tokens_total', 0.0)), 0.0)
                total_tokens = max(total_recomputed + total_reused, 1.0)
                engine_summary.update(energy_metrics)
                engine_summary['idle_baseline_power_w'] = idle_power_w
                engine_summary['idle_adjusted_gpu_energy_j'] = idle_adjusted_j
                engine_summary['gpu_joules_per_request'] = None if gpu_energy_j is None else float(gpu_energy_j) / n_req
                engine_summary['idle_adjusted_joules_per_request'] = None if idle_adjusted_j is None else float(idle_adjusted_j) / n_req
                engine_summary['gpu_joules_per_total_token'] = None if gpu_energy_j is None else float(gpu_energy_j) / total_tokens
            summary[engine.name] = engine_summary
            if args.semantic_index_diagnostics and hasattr(engine, 'semantic_index'):
                diag_file = output_dir / f'semantic_index_{engine.name}.json'
                diag_file.write_text(json.dumps(engine.semantic_index.diagnostics(), indent=2, sort_keys=True), encoding='utf-8')
                summary[engine.name]['semantic_index_diagnostics_file'] = str(diag_file)
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
    finally:
        if decision_logger is not None:
            decision_logger.close()

    if 'no_cache' in summary:
        baseline = summary['no_cache']
        baseline_j = baseline.get('gpu_joules_per_request')
        baseline_idle_j = baseline.get('idle_adjusted_joules_per_request')
        for engine_summary in summary.values():
            if not isinstance(engine_summary, dict) or 'mean_latency_ms' not in engine_summary:
                continue
            engine_summary['speedup_vs_no_cache_mean'] = baseline['mean_latency_ms'] / max(engine_summary['mean_latency_ms'], 1e-9)
            engine_summary['speedup_vs_no_cache_p95'] = baseline['p95_latency_ms'] / max(engine_summary['p95_latency_ms'], 1e-9)
            if baseline_j is not None and engine_summary.get('gpu_joules_per_request') is not None:
                engine_summary['energy_reduction_vs_no_cache_pct'] = 100.0 * (float(baseline_j) - float(engine_summary['gpu_joules_per_request'])) / max(float(baseline_j), 1e-9)
            if baseline_idle_j is not None and engine_summary.get('idle_adjusted_joules_per_request') is not None:
                engine_summary['idle_adjusted_energy_reduction_vs_no_cache_pct'] = 100.0 * (float(baseline_idle_j) - float(engine_summary['idle_adjusted_joules_per_request'])) / max(float(baseline_idle_j), 1e-9)
    summary['config'] = {key: value for key, value in vars(args).items() if not key.startswith('_')}
    summary['config']['resolved_model'] = resolve_model(args.model)
    summary['config']['config_version'] = CONFIG.version
    summary['config']['config_hash_sha256'] = CONFIG.file_hash
    summary['config']['config_path'] = str(CONFIG.path)
    summary['config']['run_manifest_file'] = str(manifest_file)
    summary['capabilities'] = capabilities or {}
    if idle_energy_report is not None:
        summary['idle_energy_baseline'] = idle_energy_report

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
    summary['config']['decision_log_file'] = str(decision_logger.path) if decision_log_enabled and decision_logger is not None else str(output_dir / str(CONFIG.get('telemetry.decision_log_name', 'policy_decisions.jsonl'))) if decision_log_enabled else None
    out_file.write_text(json.dumps(summary, indent=2))
    csv_path = None
    if args.measure_energy or args.energy_output_csv:
        csv_path = Path(args.energy_output_csv) if args.energy_output_csv else output_dir / f'{out_file.stem}_engine_summary.csv'
        csv_fields = [
            'engine', 'mean_latency_ms', 'p95_latency_ms', 'speedup_vs_no_cache_mean', 'speedup_vs_no_cache_p95',
            'hit_rate', 'gpu_energy_j', 'idle_adjusted_gpu_energy_j', 'gpu_joules_per_request',
            'idle_adjusted_joules_per_request', 'gpu_joules_per_total_token', 'energy_reduction_vs_no_cache_pct',
            'idle_adjusted_energy_reduction_vs_no_cache_pct', 'energy_source', 'idle_baseline_power_w',
            'utility_admission_enabled', 'utility_admission_checks_total', 'utility_admission_admit_total',
            'negative_utility_bypass_total', 'small_prefix_bypass_total', 'lite_fast_path_total',
            'cache_lookup_latency_total_ms', 'backend_reuse_latency_total_ms', 'kv_materialization_latency_total_ms',
            'policy_planning_latency_total_ms', 'semantic_query_latency_total_ms', 'net_latency_saved_estimate_ms_total',
        ]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()
            for engine_key, engine_summary in summary.items():
                if not isinstance(engine_summary, dict) or 'mean_latency_ms' not in engine_summary:
                    continue
                row = {'engine': engine_key}
                for field in csv_fields:
                    if field == 'engine':
                        continue
                    row[field] = engine_summary.get(field)
                writer.writerow(row)
        summary['config']['engine_summary_csv'] = str(csv_path)
        out_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved to {out_file}')
    if args.enable_policy_trace:
        print(f'Saved policy trace to {trace_file} ({len(policy_trace_rows)} rows)')
    else:
        print('Policy trace disabled. Re-run with --enable_policy_trace to emit policy_trace.jsonl.')


if __name__ == '__main__':
    main()
