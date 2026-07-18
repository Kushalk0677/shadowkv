# Gemma 4 Runtime Summary

Measured runtime baselines on RTX PRO 6000 Blackwell.

## Overall Runtime Means

| Runtime | Mean latency | Mean P95 | Throughput | GPU energy/cell | Cache evidence tokens |
|---|---:|---:|---:|---:|---:|
| vLLM APC | 25.93 ms | 32.43 ms | 60.43 req/s | 3344.96 J | 1,448,960 |
| SGLang RadixAttention | 47.51 ms | 54.97 ms | 28.09 req/s | 5496.17 J | 1,629,265 |
| LMCache + vLLM | 31.97 ms | 39.70 ms | 47.96 req/s | 4472.60 J | 66,560 |

## Model-Level Mean Latency

| Model | vLLM APC | LMCache + vLLM | SGLang RadixAttention |
|---|---:|---:|---:|
| `google/gemma-4-E2B-it` | 8.45 ms | 11.01 ms | 21.20 ms |
| `google/gemma-4-E4B-it` | 12.24 ms | 15.43 ms | 27.15 ms |
| `google/gemma-4-12B-it` | 24.82 ms | 29.73 ms | 43.90 ms |
| `google/gemma-4-26B-A4B-it` | 26.92 ms | 32.93 ms | 44.27 ms |
| `google/gemma-4-31B-it` | 57.20 ms | 70.76 ms | 101.03 ms |

## Matched Comparison Against vLLM APC

- LMCache + vLLM: 24.6% higher mean latency, 24.2% higher P95, and 39.5% higher GPU energy; 0/50 mean-latency wins.
- SGLang RadixAttention: 100.6% higher mean latency, 86.2% higher P95, and 70.2% higher GPU energy; 0/50 mean-latency wins.

## Caveats

- One run per cell; use as measured matrix evidence, not replicated confidence intervals.
- Runtime builds and memory settings differ across systems.
- This is a runtime-system comparison, not a controlled kernel-only comparison.
- No no-cache or MeritKV arm is included in this package.
