# Gemma 4 vLLM APC Runtime Results

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
| `google/gemma-4-E2B-it` | 10 | 8.45 ms | 10.70 ms | 121.13 req/s | 299,136 |
| `google/gemma-4-E4B-it` | 10 | 12.24 ms | 14.64 ms | 83.41 req/s | 299,136 |
| `google/gemma-4-12B-it` | 10 | 24.82 ms | 31.22 ms | 41.66 req/s | 252,480 |
| `google/gemma-4-26B-A4B-it` | 10 | 26.92 ms | 32.60 ms | 37.76 req/s | 299,136 |
| `google/gemma-4-31B-it` | 10 | 57.20 ms | 72.98 ms | 18.20 req/s | 299,072 |

## Caveat

These are runtime-system baselines only. This package does not include a no-cache arm or MeritKV admission-policy arm.
