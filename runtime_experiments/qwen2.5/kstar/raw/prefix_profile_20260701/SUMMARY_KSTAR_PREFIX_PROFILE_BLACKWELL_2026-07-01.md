# Blackwell k* Prefix-Length Profile

Rows: `18`

## Cached-Token Observability

Response-level cached tokens stayed `0` because this vLLM `/v1/completions` path does not populate `usage.prompt_tokens_details.cached_tokens`. The cache-hit evidence below is the vLLM `/metrics` delta, primarily `vllm:prompt_tokens_cached_total` and `vllm:prompt_tokens_by_source_total{source="local_cache_hit"}`. That is why the table separates `Response cached tokens` from `Metrics cached tokens`.

| Model | Inferred k* tokens | Positive points |
| --- | ---: | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` | `16` | 5 |
| `Qwen/Qwen2.5-7B-Instruct` | `16` | 4 |
| `Qwen/Qwen2.5-32B-Instruct` | `16` | 5 |

## Prefix Points

| Model | Prefix target | Prefix actual | Store median ms | Probe median ms | Median saving ms | Response cached tokens | Metrics cached tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` | 8 | 8 | 7.27 | 7.17 | 0.10 | 0.0 | 0.0 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 16 | 16 | 7.18 | 7.17 | 0.01 | 0.0 | 16.0 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 32 | 32 | 7.05 | 7.02 | 0.03 | 0.0 | 32.0 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 48 | 48 | 7.02 | 6.98 | 0.04 | 0.0 | 48.0 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 64 | 64 | 7.27 | 7.07 | 0.20 | 0.0 | 64.0 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 96 | 96 | 7.43 | 7.20 | 0.23 | 0.0 | 96.0 |
| `Qwen/Qwen2.5-7B-Instruct` | 8 | 8 | 14.85 | 14.60 | 0.26 | 0.0 | 0.0 |
| `Qwen/Qwen2.5-7B-Instruct` | 16 | 16 | 14.69 | 14.62 | 0.07 | 0.0 | 16.0 |
| `Qwen/Qwen2.5-7B-Instruct` | 32 | 32 | 14.77 | 14.83 | -0.06 | 0.0 | 32.0 |
| `Qwen/Qwen2.5-7B-Instruct` | 48 | 48 | 14.87 | 14.73 | 0.14 | 0.0 | 48.0 |
| `Qwen/Qwen2.5-7B-Instruct` | 64 | 64 | 15.32 | 14.72 | 0.60 | 0.0 | 64.0 |
| `Qwen/Qwen2.5-7B-Instruct` | 96 | 96 | 15.86 | 15.29 | 0.57 | 0.0 | 96.0 |
| `Qwen/Qwen2.5-32B-Instruct` | 8 | 8 | 49.69 | 49.94 | -0.25 | 0.0 | 0.0 |
| `Qwen/Qwen2.5-32B-Instruct` | 16 | 16 | 49.73 | 49.57 | 0.15 | 0.0 | 16.0 |
| `Qwen/Qwen2.5-32B-Instruct` | 32 | 32 | 50.08 | 49.26 | 0.83 | 0.0 | 32.0 |
| `Qwen/Qwen2.5-32B-Instruct` | 48 | 48 | 49.93 | 49.53 | 0.40 | 0.0 | 48.0 |
| `Qwen/Qwen2.5-32B-Instruct` | 64 | 64 | 52.50 | 49.54 | 2.96 | 0.0 | 64.0 |
| `Qwen/Qwen2.5-32B-Instruct` | 96 | 96 | 54.08 | 49.26 | 4.82 | 0.0 | 96.0 |

