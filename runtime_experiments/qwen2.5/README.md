# Qwen2.5 Runtime Experiments

This folder contains the current Qwen2.5 runtime benchmark results.

## Hardware

NVIDIA RTX PRO 6000 Blackwell.

## Models

Qwen2.5 1.5B, 3B, 7B, 14B, and 32B appear in the curated tables. The included vLLM and SGLang run files are for Qwen2.5-32B. The k-star prefix profile includes Qwen2.5-1.5B, Qwen2.5-7B, and Qwen2.5-32B.

## Layout

| Path | Contents |
|---|---|
| `sglang/results.csv` | Curated SGLang table. |
| `sglang/raw/q32b_full_20260608/` | Full SGLang/LMCache Qwen2.5-32B run files. |
| `vllm/results.csv` | Curated vLLM table. |
| `vllm/raw/q32b_5rep_20260701/` | 5-replicate vLLM Qwen2.5-32B aggregate and available benchmark JSONs. |
| `vllm/raw/q32b_20260603/` | Earlier full vLLM Qwen2.5-32B run files. |
| `lmcache/results.csv` | Curated LMCache subset used in the SGLang comparison. |
| `kstar/raw/prefix_profile_20260701/` | Primary k-star prefix-length profile. |
| `kstar/raw/response_usage_probe_20260701/` | Response-usage probe for the k-star run. |
| `kstar/raw/run_logs_20260701/` | Logs for the k-star and vLLM runtime runs. |
| `summary.md` | Cross-runtime Qwen2.5 summary. |

## Curated Tables

| File | Rows | Notes |
|---|---:|---|
| `sglang/results.csv` | 290 | Presentation table for SGLang RadixAttention, RadixAttention + MeritKV, and LMCache. |
| `vllm/results.csv` | 270 | Presentation table for vLLM no-cache, APC, and APC + MeritKV. |
| `lmcache/results.csv` | 100 | LMCache subset used for the SGLang comparison. |

## Notes

- The curated CSVs are the primary public-facing tables.
- The `raw/` subfolders keep the run files needed to audit or regenerate selected values without cluttering the top-level runtime folders.
- The July 1 vLLM aggregate contains all five replicates. The June 3 vLLM run is retained as an additional full run tree.

