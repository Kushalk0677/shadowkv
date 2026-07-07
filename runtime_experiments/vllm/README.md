# vLLM Automatic Prefix Caching Experiments

This table compares vLLM no-cache, vLLM Automatic Prefix Caching, and a MeritKV policy overlay on top of APC.

## Engines

| Engine | Description |
|--------|-------------|
| `vllm_no_cache` | Baseline vLLM without prefix caching |
| `vllm_apc` | vLLM Automatic Prefix Caching |
| `vllm_apc_shadowkv_plus` | MeritKV admission overlay on vLLM APC |

## File

- `results.csv` has 270 rows covering 5 model sizes across 10 datasets and 2 prompt modes.

## Key Columns

- `mean_latency_ms` - mean request latency
- `speedup_vs_no_cache_pct` - speedup over the no-cache baseline
- `gpu_energy_j` - total GPU energy, where available

## Notes

- Measurements were prepared on an NVIDIA RTX PRO 6000 Blackwell environment.
- 7B and 32B measurements use the full ten-dataset, two-mode sweep.
- The complete table covers Qwen2.5 model sizes 1.5B through 32B.
