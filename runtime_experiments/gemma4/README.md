# Gemma 4 Runtime Experiments

This folder contains the Gemma 4 Blackwell runtime baseline matrix from `shadowkv_gemma4_blackwell_runtime_matrix_2026-07-17.zip`.

## Status

Populated from a complete 150-cell runtime matrix: 5 Gemma 4 model variants x 5 datasets x 2 prompt modes x 3 runtime systems. Each cell used 256 requests, seed 42, temperature 0, and one output token. The archive audit reports 38,400 total requests, zero request failures, NVML energy for all cells, and runtime-native cache evidence.

## Layout

| Path | Contents |
|---|---|
| `vllm/` | vLLM APC curated table, raw result JSONs and metadata. |
| `sglang/` | SGLang RadixAttention curated table, raw result JSONs and metadata. |
| `lmcache/` | LMCache + vLLM curated table, raw result JSONs and metadata. |
| `analysis/` | Cross-runtime aggregate tables, matched comparisons, audit, block plan, and source report. |
| `kstar/` | Reserved for Gemma 4 k-star prefix profile results; no Gemma 4 k-star run is included in this archive. |

## Important Scope Note

This is a runtime-baseline matrix. It compares vLLM APC, SGLang RadixAttention, and LMCache + vLLM. It does not include a no-cache arm and does not include MeritKV admission-policy results.
