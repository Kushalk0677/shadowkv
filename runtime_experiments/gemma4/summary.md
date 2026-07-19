# Gemma 4 Blackwell Runtime Results

Measured runtime baselines on NVIDIA RTX PRO 6000 Blackwell.

## Hardware

NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM.

## Models

| Model | Params |
|-------|:-----:|
| Gemma 4 E2B | 2.3B |
| Gemma 4 E4B | 4B |
| Gemma 4 12B | 12B |
| Gemma 4 26B-A4B | 26B |
| Gemma 4 31B | 31B |

## Layout

| Path | Contents |
|------|----------|
| `vllm/results.csv` | vLLM no-cache, APC, and APC + MeritKV results. |
| `sglang/results.csv` | SGLang no-cache, RadixAttention, and RadixAttention + MeritKV results. |
| `lmcache/results.csv` | LMCache no-cache, LMCache + vLLM, and LMCache + MeritKV results. |
| `*/raw/full/rep_*` | Raw per-cell benchmark JSONs for five measured seeds. |
| `*/raw/aggregate_*` | Runtime-specific aggregate files from the run package. |

## Coverage

| Runtime | Models | Datasets | Modes | Engines | Seeds | CSV Rows | Benchmark JSONs |
|---------|:-----:|:--------:|:----:|:-------:|:----:|:--------:|:---------------:|
| vLLM | 5 | 5 | 2 | 3 | 5 | 750 | 750 |
| SGLang | 5 | 5 | 2 | 3 | 5 | 750 | 750 |
| LMCache | 5 | 5 | 2 | 3 | 5 | 750 | 750 |

Each runtime folder also includes aggregate JSON/CSV files and MeritKV admission-tuning reports, so total JSON file counts are larger than the benchmark-cell counts.

## Notes

- 256 requests per cell, temperature 0, one output token.
- NVML energy measurements are included in raw JSONs and curated CSVs.
- The no-cache arm disables runtime caching for the corresponding runtime family.
- MeritKV arms use selective admission with semantic reuse disabled.
- `latency_reduction_vs_no_cache_pct` is `100 * (1 - mean_latency / no_cache_mean_latency)` for the matched model, dataset, mode, and seed cell.
