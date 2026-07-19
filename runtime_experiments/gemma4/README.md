# Gemma 4 Blackwell Runtime Experiments

This folder contains measured Gemma 4 Blackwell runtime benchmark results with no-cache, native runtime-cache, and MeritKV admission-policy arms across five seeds.

## Contents

| Path | Contents |
|------|----------|
| `vllm/results.csv` | Curated vLLM rows for no-cache, vLLM APC, and vLLM APC + MeritKV. |
| `sglang/results.csv` | Curated SGLang rows for no-cache, RadixAttention, and RadixAttention + MeritKV. |
| `lmcache/results.csv` | Curated LMCache + vLLM rows for no-cache, LMCache, and LMCache + MeritKV. |
| `vllm/raw/full/rep_*` | Per-cell benchmark JSONs for all five seeds. |
| `sglang/raw/full/rep_*` | Per-cell benchmark JSONs for all five seeds. |
| `lmcache/raw/full/rep_*` | Per-cell benchmark JSONs for all five seeds. |
| `run_metadata.json` | Hardware, model, dataset, seed, and measurement protocol metadata. |
| `summary.md` | Cross-runtime summary and coverage notes. |

## Models

Gemma 4 E2B, E4B, 12B, 26B-A4B, and 31B.

## Datasets

`ag_news`, `daily_dialog`, `dolly`, `samsum`, and `xsum`.

## Modes

`rag` and `templated`.

## Seeds

`42`, `123`, `456`, `789`, and `999`.

## Metric Note

The curated CSV column `latency_reduction_vs_no_cache_pct` is computed as:

`100 * (1 - mean_latency_ms / no_cache_mean_latency_ms)`

It is a latency-reduction percentage, not multiplicative throughput speedup.
