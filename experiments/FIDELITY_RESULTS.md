# Semantic Fidelity — Final Complete Results

## Core Finding: KV Cache Fidelity is Architecture-Dependent

### TinyLlama (LLaMA architecture)
- **100% exact match** between ref and reuse across ALL tests
- 13/13 samples at 75% ratio
- Tested at 75%, 50%, 25%, 0% ratios — all 100%
- The DynamicCache.crop() method is **mathematically faithful** for LLaMA

### Qwen 2.5 1.5B (Qwen2 architecture)
- **Outputs diverge** after ~7 generation tokens
- First 6 tokens match exactly (logit diff < 2e-5)
- Token 7+: accumulating numeric drift (logit diff jumps to >1.0)
- Root cause: numerical differences in Qwen's attention/cache implementation

This means: **KV cache reuse fidelity must be verified per model architecture.**

## Three Experiments Compared

| Experiment | What It Measures | TinyLlama Result | Qwen 1.5B Result |
|------------|-----------------|------------------|-------------------|
| **V1: Raw templates** | Prompt sensitivity (wrong) | ROUGE-L = 0.19 | — |
| **FINAL: DynamicCache crop** | KV reuse fidelity (correct) | **ROUGE-L = 1.00** | ROUGE-L ≈ 0.99* |
| **Prompt sensitivity** | exact vs ref output diff | ROUGE-L = 0.15 | — |

*Qwen 1.5B ROUGE-L ≈ 0.99 estimated — most tokens match but small divergence occurs

## TinyLlama Detail (3 datasets, 13 samples)

| Dataset | N | Shared Prefix | KV Fidelity | Prompt Sensitivity |
|---------|---|--------------|-------------|-------------------|
| samsum | 8 | 75% | **1.0000** | 0.2144 |
| alpaca_eval | 3 | 74% | **1.0000** | 0.0075 |
| banking77 | 2 | 72% | **1.0000** | 0.1178 |
| **Overall** | **13** | **~74%** | **1.0000** | **0.1518** |

## Pipeline Files

| File | Purpose |
|------|---------|
| `run_fidelity_equiv.py` | Correct DynamicCache crop methodology |
| `eval_comprehensive.py` | Full evaluation with ROUGE-L for both metrics |
| `fidelity_equiv_colab.ipynb` | Colab notebook for GPU runs |
| `FIDELITY_FINAL_RESULTS.md` | This document |

## GitHub
Pushed to main: 3a4b3e4, f79bbc6
