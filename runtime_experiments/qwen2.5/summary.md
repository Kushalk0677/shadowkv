# Cross-Runtime Summary

This summary covers the curated runtime tables and the included run files for vLLM, SGLang/LMCache, and k-star.

## Curated Table Coverage

| Dataset | Rows | Models | Engines | Dataset count |
|---|---:|---:|---:|---:|
| SGLang | 290 | 5 | 3 | 10 |
| vLLM | 270 | 5 | 3 | 10 |
| LMCache | 100 | 5 | 1 | 10 |

## SGLang: MeritKV Speedup vs LMCache Baseline

Mean across the curated SGLang table:

| Model | MeritKV vs LMCache |
|---|---:|
| Qwen2.5-1.54B | +7.2% |
| Qwen2.5-3.09B | +8.4% |
| Qwen2.5-7.61B | +15.7% |
| Qwen2.5-14.7B | +12.7% |
| Qwen2.5-32.5B | +2.7% |

## SGLang: MeritKV Speedup vs Native RadixAttention

Mean across the curated SGLang table:

| Model | MeritKV vs RadixAttention |
|---|---:|
| Qwen2.5-1.54B | -1.1% |
| Qwen2.5-3.09B | -0.8% |
| Qwen2.5-7.61B | +2.9% |
| Qwen2.5-14.7B | +1.7% |
| Qwen2.5-32.5B | +3.7% |

## vLLM Qwen2.5-32B 5-Replicate Aggregate

The July 1 vLLM aggregate is in `vllm/raw/q32b_5rep_20260701/` and contains 150 rows: 5 reps x 10 dataset/mode cells x 3 engines.

| Engine | Jobs | Mean ms | P95 ms | Throughput rps | Idle J/request | Metrics cached tokens |
|---|---:|---:|---:|---:|---:|---:|
| vLLM APC | 50 | 59.37 | 73.62 | 17.25 | 26.63 | 1,472,032 |
| vLLM APC + MeritKV | 50 | 59.62 | 73.93 | 17.17 | 26.59 | 1,470,400 |
| vLLM no cache | 50 | 72.84 | 89.96 | 14.21 | 36.32 | 0 |

Paired deltas from the run summary:

| Comparison | Mean latency delta | P95 delta | Throughput delta | Idle J/request delta |
|---|---:|---:|---:|---:|
| APC + MeritKV vs APC | +0.44% | +0.59% | -0.42% | -0.22% |
| APC + MeritKV vs no cache | -17.59% | -18.03% | +21.56% | -27.26% |
| APC vs no cache | -17.95% | -18.42% | +22.10% | -27.09% |

## SGLang/LMCache Qwen2.5-32B Run

The June 8 SGLang/LMCache tree is in `sglang/raw/q32b_full_20260608/`.

| Baseline | Mean latency ms | P95 latency ms | Throughput rps | Cached tokens total | Cached tokens mean | Idle-adjusted J/request |
|---|---:|---:|---:|---:|---:|---:|
| sglang_radix_attention | 99.09 | 112.87 | 10.19 | 351,688 | 137.38 | 44.98 |
| lmcache | 99.45 | 114.83 | 10.15 | 359,986 | 140.62 | 44.79 |

## k-star Prefix Profile

The primary Blackwell prefix profile is in `kstar/raw/prefix_profile_20260701/`.

| Model | Inferred k* tokens | Positive points |
|---|---:|---:|
| Qwen/Qwen2.5-1.5B-Instruct | 16 | 5 |
| Qwen/Qwen2.5-7B-Instruct | 16 | 4 |
| Qwen/Qwen2.5-32B-Instruct | 16 | 5 |

Response-level cached-token fields stayed at zero for this vLLM `/v1/completions` path. The cache-hit signal comes from the vLLM metrics deltas in the k-star CSV/JSON files.


