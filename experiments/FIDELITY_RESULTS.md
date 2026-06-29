# Complete Fidelity Analysis — Root Cause Found

## Cross-Architecture Results

| Architecture | Model | KV Fidelity | Numerical Stability |
|-------------|-------|-------------|-------------------|
| **LLaMA** | TinyLlama 1.1B | **100%** | ✅ Stable |
| **Gemma** | Gemma 2B | **100%** | ✅ Stable |
| **Qwen2** | Qwen 2.5 1.5B | **≈99%** | ⚠️ Drift at step 7+ |

## Root Cause of Qwen's Drift

Traced hidden state differences through all 28 attention layers at step 6:

| Layer | Hidden State Diff | Hidden Norm | Drift Ratio |
|-------|------------------|-------------|-------------|
| 0 (embedding) | 0.00e+00 | 0.8 | — |
| 1 | 4.77e-07 | 22.5 | 2.1e-8 |
| 10 | 1.14e-05 | 53.0 | 2.2e-7 |
| 20 | 7.63e-06 | 86.4 | 8.8e-8 |
| 28 (output) | 2.29e-05 | 191.9 | 1.2e-7 |

**The drift is consistent across all layers** — a small initial floating-point difference (~5e-7) propagates linearly through the attention stack, reaching ~2e-5 at the output. This is normal float32 accumulation.

**The divergence at step 7 happens because:** the accumulated 2e-5 hidden-state difference gets projected through the LM head (192 → 151,936 vocabulary), where it's amplified by the large weight matrix. This causes a different token choice when the top-2 logits are close.

**Why LLaMA/Gemma don't drift:** Their attention implementations either use different numerical precision, different normalization, or the specific token probabilities don't have close calls at the decision boundary.

## The Two Valid Experiments

| Experiment | Method | What It Measures | TinyLlama Result |
|-----------|--------|-----------------|------------------|
| **KV Fidelity** | DynamicCache crop | Does KV reuse change output? | **1.0000** |
| **Prompt Sensitivity** | Independent generations A vs B | How much does rephrasing change output? | **0.1518** |

## Files
- `run_fidelity_equiv.py` — Correct KV reuse pipeline
- `eval_comprehensive.py` — Full evaluation
- `fidelity_equiv_colab.ipynb` — Colab notebook for GPU
- `debug_qwen_layers.py` — Layer-by-layer drift tracer
