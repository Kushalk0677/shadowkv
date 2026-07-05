# vLLM Automatic Prefix Caching Experiments

This table compares vLLM no-cache, vLLM Automatic Prefix Caching, and a ShadowKV++ policy overlay on top of APC.

## Engines

| Engine | Description |
|--------|-------------|
| `vllm_no_cache` | Baseline vLLM without prefix caching |
| `vllm_apc` | vLLM Automatic Prefix Caching |
| `vllm_apc_shadowkv_plus` | ShadowKV++ admission overlay on vLLM APC |

## File

- `results.csv` has 270 rows.

The table is not a full 5 x 3 x 10 x 2 Cartesian product. The 32B measurements are the primary measured anchor rows; smaller model sizes and missing dataset cells are scaled as noted below.

## Key Columns

- `mean_latency_ms` - mean request latency
- `speedup_vs_no_cache_pct` - speedup over the no-cache baseline
- `gpu_energy_j` - total GPU energy, where available

## Notes

- 32B measurements are from single runs.
- Smaller model sizes from 1.5B to 14B are scaled from the 32B baseline using SGLang model-size ratios.
- Missing datasets are scaled from the nearest measured dataset by token length.
- Measurements were prepared for an NVIDIA RTX PRO 6000 Blackwell environment.
