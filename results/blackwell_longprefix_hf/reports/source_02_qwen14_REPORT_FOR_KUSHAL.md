# ShadowKV++ Long-Prefix HF Results: Qwen2.5-14B

Run date: 2026-07-10 to 2026-07-11 (America/New_York)  
Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM  
Driver/CUDA: 580.119.02 / CUDA 13.0  
Runtime: `shadowkv-hf-blackwell:20260706`, PyTorch 2.11.0+cu130, Transformers 5.10.2

## Executive result

The Qwen2.5-14B targeted HF sweep completed all 30 cells with zero job failures and no external GPU-workload abort. ShadowKV++ executed exact long-scaffold reuse in every dataset cell: 127 of 128 requests reused 128 prefix tokens, for a 99.21875% hit rate and 16,256 reused prefix tokens per cell. No reuse fallback or wasted-compute event was recorded.

Across ten datasets, ShadowKV++ produced a mean per-cell latency speedup of 1.046x and a P95 speedup of 1.009x versus no-cache. Mean GPU energy per request fell by 8.6%, or 10.5% after idle adjustment. Native `shadow_kv` performed no reuse and remained at parity with no-cache.

| Engine | Mean latency | P95 latency | Mean cell speedup | P95 cell speedup | GPU energy reduction | Idle-adjusted energy reduction |
|---|---:|---:|---:|---:|---:|---:|
| no_cache | 72.502 ms | 78.227 ms | 1.000x | 1.000x | baseline | baseline |
| shadow_kv | 72.502 ms | 78.258 ms | 1.000x | 1.000x | -0.1% | +0.5% |
| shadow_kv_plus | 69.683 ms | 78.234 ms | 1.046x | 1.009x | +8.6% | +10.5% |

## Protocol

- Model: `Qwen/Qwen2.5-14B-Instruct`
- Datasets: AG News, AlpacaEval, Banking77, CNN/DailyMail, DailyDialog, Dolly, OASST1, SAMSum, UltraChat, XSum
- Mode: semantic workload generation with common long scaffold
- Engines: `no_cache`, `shadow_kv`, `shadow_kv_plus`
- Requests: 128 per cell
- Seed: 42
- Shared-prefix repeats: 4
- Cached prefix cap: 128 tokens
- Energy: NVML total and idle-adjusted energy enabled
- Policy traces: enabled
- Arrival simulation: enabled, 50 ms mean inter-arrival

Before the full sweep, a 16-request AG News smoke passed all three engines and confirmed 15 of 16 exact-scaffold reuse successes.

## Reuse-path validation

Every full-sweep ShadowKV++ cell reported:

- `path_reading=exact_scaffold_only`
- `reuse_successes=127`
- `fast_exact_path_hits=127`
- `reused_prefix_tokens_total=16256`
- `hit_rate=0.9921875`
- `wasted_compute_ratio=0.0`
- `semantic_partial_hits=0`
- `semantic_quality_divergence_events=0`
- backend fallbacks: 0

This validates real KV reuse through the exact long-scaffold path. It is not evidence for approximate semantic-partial reuse because no semantic-partial path executed.

## Dataset results

| Dataset | Mean speedup | P95 speedup | GPU energy reduction |
|---|---:|---:|---:|
| AG News | 1.087x | 1.086x | 11.9% |
| AlpacaEval | 1.087x | 1.096x | 12.0% |
| Banking77 | 1.082x | 1.085x | 12.2% |
| CNN/DailyMail | 0.987x | 0.933x | 2.1% |
| DailyDialog | 1.067x | 0.998x | 10.4% |
| Dolly | 1.053x | 0.969x | 9.7% |
| OASST1 | 1.053x | 0.934x | 11.0% |
| SAMSum | 1.035x | 1.030x | 8.9% |
| UltraChat | 1.007x | 1.027x | 5.0% |
| XSum | 1.001x | 0.935x | 2.6% |

## Anomaly audit

The run produced the expected 30 engine rows, 20 paired comparisons, 20 reuse-path rows, and 30 result JSON files. All requested latency, energy, and reuse metrics are populated. The run log reports `failed=0/30`; no traceback, OOM, external-workload abort, or nonzero backend fallback was found.

The mean-latency result is positive in nine of ten datasets. CNN/DailyMail is below parity at 0.987x. P95 is more variable: five datasets are below parity, with CNN/DailyMail, OASST1, and XSum near 0.93x. This tail result should not be hidden by the positive mean.

## Interpretation and limitations

The run establishes that exact 128-token scaffold reuse works reliably on Qwen2.5-14B and improves average latency and energy under this workload. It does not establish a uniform tail-latency improvement. A fixed 128-token reused prefix removes a different fraction of each dataset's prefill work, and the current one-seed, one-run-per-cell design cannot separate small effects from run-order or system noise.

The engines ran in fixed order (`no_cache`, `shadow_kv`, `shadow_kv_plus`). Before making a final performance claim, repeat at least three times with randomized engine order and report paired confidence intervals, especially for P95.

The patched source includes the previously validated long-scaffold admission fix, mutable-cache isolation, current Hugging Face dataset IDs, and regression tests. The source test suite passed `92 passed, 1 skipped` before this family of runs.

Production `qwen36-27b-fp8-vllm` was restored after the sweep and `/v1/models` returned HTTP 200 with model ID `qwen36-27b-fp8`.

