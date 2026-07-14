# ShadowKV++ Long-Prefix HF Results: Qwen2.5-32B

Run date: 2026-07-11 (America/New_York)  
Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM  
Driver/CUDA: 580.119.02 / CUDA 13.0  
Runtime: `shadowkv-hf-blackwell:20260706`, PyTorch 2.11.0+cu130, Transformers 5.10.2

## Executive result

The clean Qwen2.5-32B HF sweep completed all 30 cells with zero job failures and no external GPU-workload abort. ShadowKV++ executed exact long-scaffold reuse in every dataset cell: 127 of 128 requests reused 128 prefix tokens, for a 99.21875% hit rate and 16,256 reused prefix tokens per cell. No reuse fallback or wasted-compute event was recorded.

Across ten datasets, ShadowKV++ produced a mean per-cell latency speedup of 1.109x and a P95 speedup of 1.094x versus no-cache. Mean GPU energy per request fell by 11.1%, or 12.0% after idle adjustment. ShadowKV++ was above latency parity on both mean and P95 in all ten datasets. Native `shadow_kv` performed no reuse and remained effectively at parity with no-cache.

| Engine | Mean latency | P95 latency | Mean cell speedup | P95 cell speedup | GPU energy reduction | Idle-adjusted energy reduction |
|---|---:|---:|---:|---:|---:|---:|
| no_cache | 153.942 ms | 171.681 ms | 1.000x | 1.000x | baseline | baseline |
| shadow_kv | 154.206 ms | 172.045 ms | 0.998x | 0.998x | -0.2% | +0.7% |
| shadow_kv_plus | 139.349 ms | 158.413 ms | 1.109x | 1.094x | +11.1% | +12.0% |

## Protocol

- Model: `Qwen/Qwen2.5-32B-Instruct`
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

Before the full sweep, a fresh 16-request AG News smoke passed all three engines and confirmed 15 of 16 exact-scaffold reuse successes.

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
| AG News | 1.127x | 1.151x | 12.3% |
| AlpacaEval | 1.135x | 1.156x | 13.7% |
| Banking77 | 1.132x | 1.160x | 13.4% |
| CNN/DailyMail | 1.057x | 1.007x | 5.8% |
| DailyDialog | 1.126x | 1.111x | 13.2% |
| Dolly | 1.118x | 1.107x | 11.7% |
| OASST1 | 1.123x | 1.061x | 12.2% |
| SAMSum | 1.108x | 1.091x | 10.9% |
| UltraChat | 1.087x | 1.097x | 9.4% |
| XSum | 1.077x | 1.002x | 8.2% |

## Runtime patch required for 32B

The original harness loaded a second full Hugging Face model solely to run ShadowKV policy calibration after the measured backend was already loaded. Two FP16 Qwen2.5-32B copies consumed approximately 94 GiB and caused an OOM before cached-engine measurement.

The patch runs the unmeasured prefill calibration probes on the already-loaded measured backend and stores the resulting calibration dictionary for the engine. The probes do not retain KV entries or enter measured request metrics. A regression test asserts that `build_engine` does not invoke the model loader for calibration. The patched suite passed `93 passed, 1 skipped`.

## Anomaly audit

The clean run produced the expected 30 engine rows, 20 paired comparisons, 20 reuse-path rows, and 30 result JSON files. All requested latency, energy, and reuse metrics are populated. The run log reports `failed=0/30`; no traceback, OOM, external-workload abort, or nonzero backend fallback exists in this result root.

The weakest P95 cells are XSum at 1.002x and CNN/DailyMail at 1.007x. These are technically above parity but too close to support a strong dataset-specific tail claim from a single run.

## Interpretation and limitations

The run establishes that exact 128-token scaffold reuse works reliably on Qwen2.5-32B and produces a consistent mean-latency and energy benefit under this workload. The effect is larger and more uniform than the corresponding 7B and 14B runs, consistent with prefix prefill becoming more expensive as model size increases.

This is one seed and one execution per cell. Engines ran in fixed order (`no_cache`, `shadow_kv`, `shadow_kv_plus`). Before making a final performance claim, repeat at least three times with randomized engine order and report paired confidence intervals, especially for the near-parity XSum and CNN/DailyMail P95 cells.

The source snapshot includes the single-backend calibration patch, prior long-scaffold admission fix, mutable-cache isolation, current Hugging Face dataset IDs, and regression tests.

Production `qwen36-27b-fp8-vllm` was restored after the sweep and `/v1/models` returned HTTP 200 with model ID `qwen36-27b-fp8`.

