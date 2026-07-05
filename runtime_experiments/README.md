# Runtime Experiments: SGLang, LMCache, and vLLM

This directory contains curated runtime benchmark tables for ShadowKV++ on production LLM serving systems.

## Hardware

NVIDIA RTX PRO 6000 Blackwell.

## Models

Qwen2.5 family: 1.5B, 3B, 7B, 14B, and 32B.

## Datasets

Ten benchmark datasets are represented across the tables: AG News, Banking77, AlpacaEval, Dolly, DailyDialog, OASST1, UltraChat, SAMSum, XSum, and CNN/DailyMail.

The runtime tables use templated and RAG-style prompt modes.

## Engines

| System | Engines |
|--------|---------|
| SGLang | RadixAttention, RadixAttention + ShadowKV++, LMCache |
| vLLM | No cache, Automatic Prefix Caching, APC + ShadowKV++ |
| LMCache | LMCache without native RadixAttention |

## Key Results

**SGLang:** ShadowKV++ achieves 16.7% speedup over LMCache at 7B, with benefit increasing up to the 7B point in this table.

**vLLM:** At 32B, APC + ShadowKV++ improves over the no-cache baseline and reduces GPU energy in the measured setting.

## Data Files

| File | Rows | Notes |
|------|-----:|-------|
| `sglang/results.csv` | 290 | Mostly 5 models x 3 engines x 10 datasets x 2 modes, with some cells omitted or derived as noted in `sglang/README.md` |
| `vllm/results.csv` | 270 | 5-model table with measured 32B anchor rows and scaled/projected smaller-model rows as noted in `vllm/README.md` |
| `lmcache/results.csv` | 100 | 5 models x 1 engine x 10 datasets x 2 modes |
| `summary.md` | - | Cross-runtime summary tables |
| `build_complete_tables.py` | - | Table-regeneration helper for working copies that include raw deliverables |

The row counts are the actual CSV row counts. They should not be read as a complete Cartesian product unless the per-runtime README says so.
