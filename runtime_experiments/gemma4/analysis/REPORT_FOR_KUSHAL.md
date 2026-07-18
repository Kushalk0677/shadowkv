# Gemma 4 Blackwell Runtime Matrix

## Result

The requested runtime matrix completed successfully on first-light's NVIDIA RTX PRO 6000 Blackwell. Smoke coverage was 15/15 and the full run was 150/150 cells with zero request failures. Production `qwen36-27b-fp8-vllm` was restored and verified afterward.

The matrix covered five Gemma 4 models, five datasets, two prompt modes, and three cache runtimes at 256 requests per cell. All 38,400 measured requests have NVML energy data.

## Configuration

- Models: Gemma 4 E2B, E4B, 12B, 26B-A4B, and 31B instruction variants.
- Datasets: `daily_dialog`, `samsum`, `ag_news`, `dolly`, and `xsum`.
- Modes: `templated` and `rag`.
- Runtimes: vLLM APC, SGLang RadixAttention, and LMCache with vLLM.
- Request settings: 256 requests, seed 42, temperature 0, one output token.
- Energy: 30-second idle stabilization, 10-second idle baseline, NVML sampling.
- Schedule: one fixed randomized model/runtime block order; ten dataset/mode cells per loaded server block.
- vLLM and SGLang GPU allocation: 0.88. LMCache vLLM allocation: 0.82 with a 20 GB CPU L1 KV tier and 256-token chunks.

## Aggregate Results

| Runtime | Mean E2E latency | Mean P95 | Mean throughput | Mean GPU energy/cell | Native cache evidence |
|---|---:|---:|---:|---:|---:|
| vLLM APC | 25.93 ms | 32.43 ms | 60.43 req/s | 3,344.96 J | 1,448,960 local hit tokens |
| LMCache + vLLM | 31.97 ms | 39.70 ms | 47.96 req/s | 4,472.60 J | 66,560 external hit tokens |
| SGLang RadixAttention | 47.51 ms | 54.97 ms | 28.09 req/s | 5,496.17 J | 1,629,265 cached tokens |

Matched against vLLM APC across the same 50 model/dataset/mode cells:

- LMCache had 24.6% higher mean latency, 24.2% higher P95, and 39.5% higher GPU energy. It had 0/50 mean-latency wins.
- SGLang RadixAttention had 100.6% higher mean latency, 86.2% higher P95, and 70.2% higher GPU energy. It had 0/50 mean-latency wins.

## Model-Level Mean Latency

| Model | vLLM APC | LMCache + vLLM | SGLang RadixAttention |
|---|---:|---:|---:|
| Gemma 4 E2B | 8.45 ms | 11.01 ms | 21.20 ms |
| Gemma 4 E4B | 12.24 ms | 15.43 ms | 27.15 ms |
| Gemma 4 12B | 24.82 ms | 29.73 ms | 43.90 ms |
| Gemma 4 26B-A4B | 26.92 ms | 32.93 ms | 44.27 ms |
| Gemma 4 31B | 57.20 ms | 70.76 ms | 101.03 ms |

## Cache Verification

- vLLM APC reported positive local prefix-cache hit deltas in all 50 cells.
- SGLang reported positive cached-token totals in all 50 cells.
- LMCache reported external KV hit deltas in 40/50 cells and its five model logs contain 4,275 store events plus 260 retrieve events. Those retrieve events transferred 66,560 tokens.
- LMCache's ten AG News cells reported zero external hits. Their reusable prefixes were below the configured 256-token chunk granularity, so those cells measured LMCache overhead without an external KV retrieval.
- `cached_tokens_total=0` in the generic vLLM/LMCache response summary is not used as evidence. The authoritative fields are vLLM Prometheus local/external cache deltas and LMCache transfer logs.

## Interpretation

For these short, sequential, one-token-generation workloads, native vLLM APC is the best of the three tested systems. LMCache is functioning correctly, but CPU-tier lookup/transfer and connector overhead exceed its reuse savings here. SGLang RadixAttention records the most cached tokens, but its tested server stack is still slower than vLLM APC, showing that cached-token volume alone does not predict end-to-end latency.

This run compares three cache-enabled runtime systems. It does not include a no-cache arm, so it does not quantify the absolute benefit of APC, RadixAttention, or LMCache relative to full prefill.

## Caveats

- This is one run per cell. The block order was randomized once, but the results are not replicated estimates.
- Runtime builds differ: vLLM nightly 0.23.1 development, SGLang development build, and LMCache 0.5.1 with vLLM 0.24.0. Exact versions and image inspections are included.
- Memory settings differ because LMCache reserves an external CPU KV tier: 0.88 GPU allocation for vLLM/SGLang versus 0.82 for LMCache vLLM.
- SGLang and vLLM use their own request formatting/tokenization paths. This is a system-level comparison, not a controlled kernel-only comparison.
- The 12B server logs contain nonfatal multimodal warmup warnings. All measured text requests completed successfully.

## Included Evidence

- Raw smoke and full result JSON.
- Per-cell CSV and matched-comparison CSV.
- Model/runtime summaries and anomaly audit.
- Native server logs and LMCache store/retrieve logs.
- Hardware, image, package-version, runner, and source snapshots.
- Archived diagnostics for the failed pre-run attempts and the narrow fixes used before the successful matrix.

