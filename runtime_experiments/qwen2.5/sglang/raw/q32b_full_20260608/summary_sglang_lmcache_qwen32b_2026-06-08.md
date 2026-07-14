# SGLang + LMCache Qwen2.5-32B Full Run Summary - 2026-06-08

Result root: `/home/jade_hand/research/shadowkv/results_sglang_lmcache_q32b_full_20260608`

## Aggregate Averages

| Baseline | Mean latency ms | P95 latency ms | Throughput rps | Cached tokens total | Cached tokens mean | Idle-adjusted J/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sglang_radix_attention | 99.09 | 112.87 | 10.19 | 351688 | 137.38 | 44.98 |
| lmcache | 99.45 | 114.83 | 10.15 | 359986 | 140.62 | 44.79 |

## Per-Cell LMCache vs SGLang

| Dataset | Mode | Latency delta | Throughput delta | Cached token delta | Energy delta |
| --- | --- | ---: | ---: | ---: | ---: |
| ag_news | templated | 0.69% | -0.68% | 0 | 1.16% |
| ag_news | rag | 0.57% | -0.56% | 0 | -0.36% |
| daily_dialog | templated | 0.77% | -0.76% | 116 | 1.08% |
| daily_dialog | rag | 0.66% | -0.66% | 101 | -0.17% |
| dolly | templated | 0.65% | -0.65% | 399 | -0.42% |
| dolly | rag | 0.67% | -0.66% | 345 | -0.21% |
| samsum | templated | 0.32% | -0.32% | 1184 | -1.75% |
| samsum | rag | 0.35% | -0.35% | 1165 | -0.22% |
| xsum | templated | -0.54% | 0.54% | 2343 | -1.41% |
| xsum | rag | -0.16% | 0.16% | 2645 | -1.01% |

