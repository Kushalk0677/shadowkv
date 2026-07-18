# Gemma 4 LMCache + vLLM Runtime Results

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
| `google/gemma-4-E2B-it` | 10 | 11.01 ms | 13.66 ms | 93.27 req/s | 13,312 |
| `google/gemma-4-E4B-it` | 10 | 15.43 ms | 19.10 ms | 66.61 req/s | 13,312 |
| `google/gemma-4-12B-it` | 10 | 29.73 ms | 36.73 ms | 34.60 req/s | 13,312 |
| `google/gemma-4-26B-A4B-it` | 10 | 32.93 ms | 37.19 ms | 30.59 req/s | 13,312 |
| `google/gemma-4-31B-it` | 10 | 70.76 ms | 91.80 ms | 14.74 req/s | 13,312 |

## Caveat

These are runtime-system baselines only. This package does not include a no-cache arm or MeritKV admission-policy arm.
