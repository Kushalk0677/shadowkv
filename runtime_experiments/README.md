# Runtime Experiments: Real-World SGLang, LMCache, and vLLM Benchmarks

This directory contains runtime benchmark results for ShadowKV++ deployed on
production LLM serving systems.

## Hardware

NVIDIA RTX PRO 6000 Blackwell.

## Models

Qwen2.5 family: 1.5B, 3B, 7B, 14B, 32B parameters.

## Datasets

All ten benchmark datasets: AG News, Banking77, AlpacaEval, Dolly, DailyDialog,
OASST1, UltraChat, SAMSum, XSum, CNN/DailyMail.

Two prompt modes: templated and RAG.

## Engines

| System | Engines |
|--------|---------|
| SGLang v0.4.0.post2 | RadixAttention, RadixAttention + ShadowKV++, LMCache |
| vLLM v0.6.0 | No cache, Automatic Prefix Caching, APC + ShadowKV++ |

## Key Results

**SGLang:** ShadowKV++ achieves 16.7% speedup over LMCache at 7B, with
benefit increasing with model size up to 7B.

**vLLM:** At 32B, APC + ShadowKV++ achieves 19.0% speedup over no-cache
and reduces GPU energy by 25%.

## Data Files

| File | Description |
|------|-------------|
| `sglang/results.csv` | 290 rows — 5 models x 3 engines x 10 datasets x 2 modes |
| `vllm/results.csv` | 270 rows — 5 models x 3 engines x 10 datasets x 2 modes |
| `lmcache/results.csv` | 100 rows — 5 models x 1 engine x 10 datasets x 2 modes |
| `summary.md` | Cross-runtime summary tables |
| `build_complete_tables.py` | Script to regenerate all tables |
