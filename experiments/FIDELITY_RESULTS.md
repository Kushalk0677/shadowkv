# Semantic Fidelity — Complete Cross-Architecture Results

## KV Fidelity by Model Architecture

| Architecture | Model | Shared Prefix | KV Fidelity | Behavior |
|-------------|-------|--------------|-------------|----------|
| **LLaMA** | TinyLlama 1.1B | 75% | **1.0000** | All tokens match |
| **Gemma** | Gemma 2B | 75% | **1.0000** | All tokens match |
| **Qwen2** | Qwen 2.5 1.5B | 75% | **≈0.99** | Drift starts at token 7 |

## Key Finding

**KV cache fidelity is architecture-dependent, not model-size dependent:**

- LLaMA-family and Gemma: `DynamicCache.crop()` + suffix prefill is **mathematically faithful**
- Qwen2-family: Small numerical drift accumulates, causing divergence after ~7 generation tokens

## TinyLlama Detail (3 datasets, 13 samples)

| Dataset | N | KV Fidelity | Prompt Sensitivity |
|---------|---|-------------|-------------------|
| samsum | 8 | **1.0000** | 0.2144 |
| alpaca_eval | 3 | **1.0000** | 0.0075 |
| banking77 | 2 | **1.0000** | 0.1178 |
| **Overall** | **13** | **1.0000** | **0.1518** |

## Methodology

The correct experiment uses `DynamicCache.crop()` (transformers 5.x):
1. Prefill prompt A → DynamicCache
2. `cache.crop(shared)` → keep only shared prefix tokens
3. Prefill prompt B's suffix on cropped cache
4. Generate from combined state
5. Compare to clean generation from B

## Files

| File | Purpose |
|------|---------|
| `run_fidelity_equiv.py` | Correct pipeline |
| `eval_comprehensive.py` | Evaluation with ROUGE-L |
| `fidelity_equiv_colab.ipynb` | Colab notebook for GPU |
| `FIDELITY_RESULTS.md` | This document |
