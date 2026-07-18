# Gemma 4 SGLang RadixAttention Runtime Results

Measured on RTX PRO 6000 Blackwell as part of `shadowkv_gemma4_blackwell_runtime_matrix_2026-07-17.zip`.

## Coverage

- 5 Gemma 4 model variants.
- 5 datasets: `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`.
- 2 modes: `templated`, `rag`.
- 256 requests per cell, seed 42, temperature 0, one output token.
- Raw JSON files are under `raw/gemma4_runtime_matrix_20260717/`; full run logs are centralized in `../analysis/run_logs/`.

## Model Means

| Model | Cells | Mean latency | Mean P95 | Throughput | Cache evidence tokens |
|---|---:|---:|---:|---:|---:|
| `google/gemma-4-E2B-it` | 10 | 21.20 ms | 23.40 ms | 47.33 req/s | 325,867 |
| `google/gemma-4-E4B-it` | 10 | 27.15 ms | 30.72 ms | 37.10 req/s | 325,867 |
| `google/gemma-4-12B-it` | 10 | 43.90 ms | 49.86 ms | 22.98 req/s | 325,867 |
| `google/gemma-4-26B-A4B-it` | 10 | 44.27 ms | 54.49 ms | 23.03 req/s | 325,867 |
| `google/gemma-4-31B-it` | 10 | 101.03 ms | 116.39 ms | 10.02 req/s | 325,797 |

## Caveat

These are runtime-system baselines only. This package does not include a no-cache arm or MeritKV admission-policy arm.
