# ShadowKV++ HF Blackwell Seven-Model Long-Prefix Results

## Status

Complete and verified. The experiment produced 210/210 full benchmark cells with zero measured failures across seven models, ten datasets, and three engines.

## Main Finding

ShadowKV++ shows a clear break-even effect in this exact-prefix workload. Reuse materially improves both Gemma 4 models, but its fixed planning and KV-handling overhead dominates inference time on the smaller GPT-2, TinyLlama, Qwen2.5, and Phi-3 models.

- Gemma 4 E2B: 21.0% lower aggregate mean latency, 21.9% lower aggregate P95 latency, and 24.7% lower GPU energy per request versus no-cache.
- Gemma 4 12B: 26.6% lower aggregate mean latency, 31.4% lower aggregate P95 latency, and 28.4% lower GPU energy per request versus no-cache.
- GPT-2 through Qwen2.5-3B: ShadowKV++ is slower in all ten datasets for mean latency. The regression narrows as model compute increases, from 31.4% on GPT-2 to 5.1% on Qwen2.5-3B.
- Phi-3 Mini 4K: 10.2% higher aggregate mean latency despite successful reuse.

All ShadowKV++ cells executed the intended path: 127/128 reuse successes, 16,256 reused prefix tokens, `exact_scaffold_only`, zero waste, and zero backend fallbacks.

## Configuration

- Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB VRAM.
- Runtime: Hugging Face Transformers backend in `shadowkv-hf-blackwell:20260706`.
- Device and dtype: CUDA, float16.
- Models: GPT-2; TinyLlama 1.1B Chat; Qwen2.5 1.5B Instruct; Phi-3 Mini 4K Instruct; Qwen2.5 3B Instruct; Gemma 4 E2B IT; Gemma 4 12B IT.
- Datasets: AG News, AlpacaEval, Banking77, CNN/DailyMail, DailyDialog, Dolly, OASST1, SAMSum, UltraChat, and XSum.
- Workload: semantic mode with a common 128-token scaffold repeated four ways.
- Requests: 128 per cell.
- Seed: 42.
- Engines: `no_cache`, `shadow_kv`, and `shadow_kv_plus`.
- Energy: NVML measurement with a five-second idle baseline per cell.
- Arrival simulation: 50 ms mean inter-arrival, 500 ms maximum sleep.

## Results

Positive latency reduction means ShadowKV++ was faster. The speedup column is the arithmetic mean of the ten per-dataset speedup ratios.

| Model | No-cache mean ms | ShadowKV++ mean ms | Mean reduction | No-cache P95 ms | ShadowKV++ P95 ms | P95 reduction | Mean speedup | GPU energy reduction | Mean wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 | 3.05 | 4.00 | -31.4% | 3.72 | 4.55 | -22.4% | 0.762x | +0.1% | 0/10 |
| TinyLlama 1.1B Chat | 9.98 | 11.99 | -20.1% | 11.20 | 13.17 | -17.6% | 0.836x | -6.2% | 0/10 |
| Qwen2.5 1.5B Instruct | 12.63 | 14.78 | -17.0% | 14.10 | 16.30 | -15.6% | 0.855x | -0.2% | 0/10 |
| Qwen2.5 3B Instruct | 21.40 | 22.51 | -5.1% | 23.28 | 25.10 | -7.8% | 0.959x | +4.9% | 0/10 |
| Phi-3 Mini 4K Instruct | 27.04 | 29.79 | -10.2% | 30.37 | 33.33 | -9.8% | 0.913x | +0.2% | 0/10 |
| Gemma 4 E2B IT | 28.35 | 22.41 | +21.0% | 30.89 | 24.12 | +21.9% | 1.277x | +24.7% | 10/10 |
| Gemma 4 12B IT | 79.77 | 58.54 | +26.6% | 86.79 | 59.53 | +31.4% | 1.430x | +28.4% | 10/10 |

Idle-adjusted energy reductions were 30.6% for Gemma 4 E2B and 30.9% for Gemma 4 12B. The smaller-model idle-adjusted values are less stable because each request is short and fixed measurement overhead is a larger fraction of the cell.

## Interpretation

The experiment confirms that correct cache reuse alone is not sufficient for a latency win. ShadowKV++ reused the same 127 prefixes in every dataset for every model, yet the result changed with model/runtime cost:

1. For very small models, no-cache forward passes are only about 3-21 ms. ShadowKV++ bookkeeping and external-KV handling cost more than the avoided prefill compute.
2. Qwen2.5-3B is close to break-even at 0.959x mean speedup, suggesting the crossover is near this compute range for conventional text-only models under this setup.
3. Gemma 4 E2B crosses the threshold decisively despite its E2B label. Its effective-parameter architecture does not imply GPT-2-class runtime cost; measured no-cache latency was 28.35 ms.
4. Gemma 4 12B produces the strongest result because the avoided prefix compute is large relative to the fixed reuse overhead.

The native `shadow_kv` engine remained close to no-cache latency and reported no successful reuse. The gains and regressions above therefore come from the enforced ShadowKV++ exact-reuse path rather than generic run ordering alone.

## Validation And Anomalies

- Aggregate rows: 210 expected, 210 present.
- Result JSON files: 210 expected, 210 present.
- Per model: 30 cells each.
- ShadowKV++ comparison cells: 70.
- Reuse invariant: 127/128 successes and 16,256 reused tokens in all 70 cells.
- Reuse path: `exact_scaffold_only` in all 70 cells.
- Waste ratio: 0.0 in all 70 cells.
- Backend fallback count: 0.
- External GPU workload aborts: 0.
- Production restoration: `qwen36-27b-fp8-vllm` healthy and serving `Qwen/Qwen3.6-27B-FP8` after completion.

The six-model wrapper was replaced on disk while it was still running, which caused an unexpected-EOF shell error after all 180 measured cells and aggregate CSVs had already completed. Its EXIT trap restored production successfully. This was an orchestration-only event; it is preserved in the logs and metadata. Gemma 4 E2B was downloaded after that block and run with a separate immutable wrapper.

ComfyUI was not running, so its queue endpoint returned connection refused during preflight. The guard treated the absent service as an empty queue. No ComfyUI or game process appeared during either measured block.

## Caveats

- This is one seed and one execution block per model, not a randomized multi-repetition study.
- The workload deliberately provides an exact reusable 128-token scaffold. It establishes the runtime break-even curve under successful reuse; it does not measure semantic-match quality or natural prefix frequency.
- Gemma 4 E2B ran immediately after the other six models, using the same image, hardware, settings, and result schema, but in a separate clean block because its checkpoint had to be downloaded.
- Arithmetic means of per-dataset ratios differ slightly from ratios calculated from aggregate mean latencies. Both are included in the artifacts.

## Recommended Follow-Up

Use these data as a calibration result, not a universal small-model claim. The next targeted test should sweep prefix length for Qwen2.5-3B, Phi-3 Mini, and Gemma 4 E2B to estimate architecture-specific break-even thresholds. A three-repetition randomized run is warranted before publishing the Gemma 4 E2B and 12B latency percentages as final performance claims.

