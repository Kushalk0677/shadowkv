# Process-Isolated Baseline Comparison

## Engine Name Aliases

Raw artifacts keep the stable engine IDs used during execution. Public-facing text maps them as follows:

| Engine ID | Display name |
|---|---|
| `shadow_kv_plus` | MeritKV |
| `shadow_kv` | MeritKV-Sem |
| `shadow_kv_plus_lite` | MeritKV-Lite |


Process-isolated benchmark results comparing MeritKV (`shadow_kv_plus`)
against all baselines on P100.

## Experimental Setup

- Models: Gemma-2B, Qwen2.5-1.5B, TinyLlama-1.1B, GPT-2, Phi-3-mini
- Hardware: Tesla P100-PCIE-12GB, HF backend, float16, 64 requests per cell
- 5 models x 10 datasets x 3 modes x 3 seeds = 360 cells per engine
- Engines: no_cache, MeritKV-Sem (`shadow_kv`), MeritKV (`shadow_kv_plus`), MeritKV-Lite (`shadow_kv_plus_lite`), reactive_prefix_cache, greedy_prefix_cache, strict_reactive_prefix_cache

## Summary: MeritKV vs All Baselines

| Engine | Count | Mean Speedup | Mean Hit Rate | Mean Waste |
|--------|:----:|:-----------:|:------------:|:---------:|
| MeritKV (`shadow_kv_plus`) | 360 | 1.110x | 0.474 | 0.056 |
| MeritKV-Sem (`shadow_kv`) | 360 | 1.085x | 0.703 | 0.240 |
| strict_reactive_prefix_cache | 360 | 1.017x | 0.317 | 0.012 |
| no_cache | 360 | 1.000x | 0.000 | 0.000 |
| greedy_prefix_cache | 360 | 0.994x | 0.317 | 0.000 |
| reactive_prefix_cache | 360 | 0.983x | 0.317 | 0.037 |
| MeritKV-Lite (`shadow_kv_plus_lite`) | 360 | 0.920x | 0.317 | 0.000 |

## Per-Mode Speedup

| Engine | Templated | Semantic | Raw |
|--------|:---------:|:--------:|:---:|
| MeritKV (`shadow_kv_plus`) | 1.281x | 1.047x | 1.001x |
| MeritKV-Sem (`shadow_kv`) | 1.253x | 1.018x | 0.984x |
| strict_reactive_prefix_cache | 1.179x | 0.943x | 0.931x |
| no_cache | 1.000x | 1.000x | 1.000x |
| greedy_prefix_cache | 1.179x | 0.932x | 0.871x |
| reactive_prefix_cache | 1.128x | 0.922x | 0.901x |
| MeritKV-Lite (`shadow_kv_plus_lite`) | 1.089x | 0.869x | 0.801x |

## Key Findings

1. **MeritKV vs MeritKV-Sem**: MeritKV 1.110x vs MeritKV-Sem 1.085x (+2.3%) with 0.056 waste vs 0.240 waste.
2. **MeritKV vs greedy_prefix_cache**: MeritKV 1.110x vs greedy_prefix_cache 0.994x (+11.7%). MeritKV wins by 11.7%.
2. **MeritKV vs reactive_prefix_cache**: MeritKV 1.110x vs reactive_prefix_cache 0.983x (+12.9%). MeritKV wins by 12.9%.
2. **MeritKV vs strict_reactive_prefix_cache**: MeritKV 1.110x vs strict_reactive_prefix_cache 1.017x (+9.1%). MeritKV wins by 9.1%.
2. **MeritKV vs MeritKV-Lite (`shadow_kv_plus_lite`)**: MeritKV 1.110x vs MeritKV-Lite 0.920x (+20.7%). MeritKV wins by 20.7%.
