# ShadowKV++ Long-Prefix HF Results: Gemma 4 31B and 26B-A4B

Run date: 2026-07-11 (America/New_York)  
Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM  
Driver/CUDA: 580.119.02 / CUDA 13.0  
Runtime: `shadowkv-hf-blackwell:20260706`, PyTorch 2.11.0+cu130, Transformers 5.10.2

## Executive result

The clean two-model HF sweep completed all 60 cells with zero job failures and no external GPU-workload abort. ShadowKV++ executed exact long-scaffold reuse in every dataset cell: 127 of 128 requests reused 128 prefix tokens, for a 99.21875% hit rate and 16,256 reused prefix tokens per cell. No reuse fallback or wasted-compute event was recorded.

ShadowKV++ averaged 1.561x mean-latency and 1.720x P95 speedup on Gemma 4 31B, with 30.4% GPU-energy reduction. On Gemma 4 26B-A4B, it averaged 1.204x mean-latency and 1.233x P95 speedup, with 20.6% GPU-energy reduction. Both models were above mean and P95 latency parity in all ten datasets.

| Model | Engine | Mean latency | P95 latency | Mean cell speedup | P95 cell speedup | GPU energy reduction | Idle-adjusted energy reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 31B | no_cache | 177.182 ms | 196.610 ms | 1.000x | 1.000x | baseline | baseline |
| Gemma 4 31B | shadow_kv | 177.570 ms | 197.004 ms | 0.998x | 0.998x | -0.5% | +0.3% |
| Gemma 4 31B | shadow_kv_plus | 120.214 ms | 121.383 ms | 1.561x | 1.720x | +30.4% | +32.1% |
| Gemma 4 26B-A4B | no_cache | 98.037 ms | 103.759 ms | 1.000x | 1.000x | baseline | baseline |
| Gemma 4 26B-A4B | shadow_kv | 98.012 ms | 103.669 ms | 1.000x | 1.001x | -0.1% | +0.8% |
| Gemma 4 26B-A4B | shadow_kv_plus | 82.470 ms | 85.079 ms | 1.204x | 1.233x | +20.6% | +23.2% |

## Protocol

- Models: `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it`
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

A fresh six-cell AG News smoke passed all three engines for both models and confirmed 15 of 16 exact-scaffold reuse successes for each ShadowKV++ cell.

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

## Dataset ranges

Gemma 4 31B ShadowKV++ mean speedup ranged from 1.139x to 2.703x; P95 ranged from 1.145x to 2.958x. Gemma 4 26B-A4B mean speedup ranged from 1.043x to 1.595x; P95 ranged from 1.049x to 1.591x.

The largest effects occurred on CNN/DailyMail for both models. UltraChat and XSum also produced large effects. The fixed 128-token prefix removes a different fraction of each dataset's total prefill work, so this spread is plausible, but the largest cells need replication before being used as dataset-specific claims.

## Anomaly audit

The clean run produced the expected 60 engine rows, 40 paired comparisons, 40 reuse-path rows, and 60 result JSON files. All requested latency, energy, and reuse metrics are populated. The run log reports `failed=0/60`; no traceback, OOM, external-workload abort, or nonzero backend fallback exists in this result root.

All 20 ShadowKV++ comparison cells are above parity for both mean and P95 latency. The weakest cells were Gemma 26B-A4B AlpacaEval at 1.043x mean and 1.049x P95, and Gemma 31B AG News at 1.139x mean and 1.145x P95.

## Interpretation and limitations

These results establish reliable exact 128-token scaffold reuse on both dense Gemma 4 31B and MoE Gemma 4 26B-A4B. The dense 31B model benefits more strongly than A4B, consistent with each avoided prefill token requiring substantially more active computation in the dense model.

This is one seed and one execution per cell. Engines ran in fixed order (`no_cache`, `shadow_kv`, `shadow_kv_plus`). Before making final performance claims, repeat at least three times with randomized engine order and report paired confidence intervals, particularly for the largest CNN/DailyMail, UltraChat, and XSum effects.

The source snapshot includes the single-backend calibration patch required for large FP16 models, the long-scaffold admission fix, mutable-cache isolation, current Hugging Face dataset IDs, and regression tests. The patched suite passed `93 passed, 1 skipped`.

Production `qwen36-27b-fp8-vllm` was restored after the sweep and `/v1/models` returned HTTP 200 with model ID `qwen36-27b-fp8`.

