from __future__ import annotations

ENGINE_DISPLAY_NAMES = {
    'shadow_kv': 'MeritKV-Sem',
    'shadow_kv_plus': 'MeritKV',
    'shadow_kv_plus_lite': 'MeritKV-Lite',
    'shadow_kv_plus_best_latency': 'MeritKV-BestLatency',
    'shadow_kv_plus_raw_observer': 'MeritKV-RawObserver',
    'shadow_kv_plus_scaffold_only': 'MeritKV-ScaffoldOnly',
    'shadow_kv_plus_early_layer': 'MeritKV-EarlyLayer',
    'shadow_kv_plus_logit_guard': 'MeritKV-LogitGuard',
    'vllm_apc_shadowkv_plus': 'vLLM APC + MeritKV',
    'sglang_radix_attention_shadowkv_plus': 'SGLang RadixAttention + MeritKV',
    'lmcache_shadowkv_plus': 'LMCache + MeritKV',
}


def display_engine_name(engine_name: str) -> str:
    """Return the reader-facing name for a stable internal engine ID."""
    return ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)


def display_engine_names(engine_names: list[str] | tuple[str, ...]) -> dict[str, str]:
    return {name: display_engine_name(name) for name in engine_names}