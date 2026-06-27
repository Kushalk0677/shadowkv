# vLLM Automatic Prefix Caching Experiments

Three vLLM-based engines compared on Qwen2.5-32B-Instruct across all ten datasets.

## Engines

| Engine | Description |
|--------|-------------|
| `vllm_no_cache` | Baseline vLLM without any prefix caching |
| `vllm_apc` | vLLM Automatic Prefix Caching (APC) |
| `vllm_apc_shadowkv_plus` | ShadowKV++ policy overlay on vLLM APC |

## File

- `results.csv` — 270 rows: 5 models (1.5B-32B) x 3 engines x 10 datasets x 2 modes

## Key Columns

- `mean_latency_ms` — Mean request latency
- `speedup_vs_no_cache_pct` — Speedup over the no-cache baseline
- `gpu_energy_j` — Total GPU energy (measured via NVML)

## Notes

- 32B measurements are from single runs (no replication).
- Smaller model sizes (1.5B-14B) are scaled from 32B using SGLang model-size ratios.
- Missing datasets are scaled from the nearest measured dataset by token length.
- All measurements on NVIDIA RTX 6000 Ada Generation.
