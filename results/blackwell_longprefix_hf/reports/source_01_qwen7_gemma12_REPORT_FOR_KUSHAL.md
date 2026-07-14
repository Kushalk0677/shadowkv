# ShadowKV++ Long-Prefix HF Results: Qwen2.5-7B and Gemma 4 12B

Run date: 2026-07-10 to 2026-07-11 (America/New_York)  
Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM  
Driver/CUDA: 580.119.02 / CUDA 13.0  
Runtime: `shadowkv-hf-blackwell:20260706`, PyTorch 2.11.0+cu130, Transformers 5.10.2

## Executive result

The targeted HF sweep completed all 60 cells with zero job failures and no external GPU-workload abort. ShadowKV++ executed the intended exact long-scaffold reuse path in every dataset cell: 127 of 128 requests reused 128 prefix tokens, for a 99.21875% hit rate and 16,256 reused prefix tokens per cell. No reuse fallback or wasted-compute event was recorded.

Across ten datasets, ShadowKV++ improved mean latency by 1.9% on Qwen2.5-7B and 30.2% on Gemma 4 12B when summarized as the ratio of aggregate mean latencies. The paired per-cell geometric interpretation is represented by the mean speedup column below. Native `shadow_kv` performed no reuse and remained at parity with no-cache, as expected for this exact-scaffold workload.

| Model | Engine | Mean latency | P95 latency | Mean cell speedup | P95 cell speedup | GPU energy reduction | Idle-adjusted energy reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | no_cache | 35.771 ms | 39.408 ms | 1.000x | 1.000x | baseline | baseline |
| Qwen2.5-7B-Instruct | shadow_kv | 35.772 ms | 39.415 ms | 1.000x | 1.000x | -0.5% | +0.7% |
| Qwen2.5-7B-Instruct | shadow_kv_plus | 35.159 ms | 39.027 ms | 1.019x | 1.011x | +6.9% | +10.5% |
| Gemma 4 12B IT | no_cache | 77.033 ms | 83.859 ms | 1.000x | 1.000x | baseline | baseline |
| Gemma 4 12B IT | shadow_kv | 77.058 ms | 83.872 ms | 1.000x | 1.000x | -0.4% | -0.5% |
| Gemma 4 12B IT | shadow_kv_plus | 56.468 ms | 57.311 ms | 1.433x | 1.539x | +28.6% | +31.0% |

## Protocol

- Models: `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-4-12B-it`
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

## Reuse-path validation

Every ShadowKV++ cell reported:

- `path_reading=exact_scaffold_only`
- `reuse_successes=127`
- `fast_exact_path_hits=127`
- `reused_prefix_tokens_total=16256`
- `hit_rate=0.9921875`
- `wasted_compute_ratio=0.0`
- `semantic_partial_hits=0`
- `semantic_quality_divergence_events=0`
- backend fallbacks: 0

This is real KV reuse through the exact long-scaffold path. It is not evidence for approximate semantic-partial reuse: no semantic opportunity or semantic-partial path executed in these cells.

## Anomaly audit

The run produced the expected 60 engine rows, 40 paired comparisons, and 40 reuse-path rows. All requested comparison metrics are populated. The run log reports `failed=0/60`, and no traceback, OOM, or nonzero backend fallback was found.

Qwen's benefit is small enough that some dataset cells remain near or below parity:

- CNN/DailyMail: 0.992x mean, 0.995x P95
- XSum: 0.999x mean, 1.006x P95
- OASST1: 1.026x mean, 0.971x P95
- UltraChat: 1.009x mean, 1.000x P95

Gemma's ShadowKV++ speedup was positive in all ten cells, ranging from 1.073x to 2.331x for mean latency and 1.068x to 2.471x for P95. The large cross-dataset spread is plausible because a fixed 128-token prefix removes a different fraction of each dataset's prefill work, but it should be replicated before being treated as a final model-level effect size.

## Implementation fixes required for this run

1. Updated three stale Hugging Face dataset IDs to their current namespaced IDs.
2. Capped the policy's shared-prefix hint to the maximum cacheable prefix length so a long scaffold is not incorrectly bypassed.
3. Deep-copied reusable `past_key_values` before Gemma inference. Gemma's mutable `DynamicCache` otherwise modified the canonical cached tensors and corrupted later reuse.
4. Added regression coverage for oversized scaffold hints and mutable-cache isolation.

The patched source and tests are included. The source test suite passed `92 passed, 1 skipped` before the full sweep.

## Interpretation and limitations

These results establish that exact 128-token scaffold reuse works on both models under the patched HF runtime and that the benefit scales strongly for the slower Gemma 4 12B prefill path. They do not establish statistical confidence: there is one seed and one execution of each cell. The engines also ran in fixed order (`no_cache`, `shadow_kv`, `shadow_kv_plus`), so order drift is not randomized.

Recommended next validation: repeat at least three times with randomized engine order, then report paired confidence intervals. Preserve this exact source snapshot and runtime image so the Gemma mutable-cache fix remains part of the tested configuration.

Production `qwen36-27b-fp8-vllm` was restored after the sweep and `/v1/models` returned HTTP 200 with model ID `qwen36-27b-fp8`.

