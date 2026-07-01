# KV Cache Reuse Fidelity — Deep Analysis

## 1. Qwen's Failure on GPU (float16)

### Observed Behavior

On GPU (T4, float16), Qwen 2.5 1.5B shows **0.20 ROUGE-L** between ref (clean generation) and reuse (with KV cache splice). This is near-random similarity, indicating the KV reuse produces completely different outputs.

On CPU (float32), earlier tests showed ~0.99 correspondence with divergence starting at token 7.

### Root Cause: Precision �- Architecture Interaction

The failure is a **compounding effect** of two factors:

**Factor 1: Float16 precision loss in DynamicCache**
- The KV cache stores attention key/value tensors. On GPU with float16, each cache entry has only ~3.3 decimal digits of precision (vs ~7.3 for float32).
- When `cache.crop(shared)` truncates the cache and new tokens are appended, the combined cache has **inconsistent numerical precision** at the crop boundary.
- The attention scores computed across this boundary accumulate rounding errors faster than in float32.

**Factor 2: Qwen2's attention sensitivity**
- Qwen2 uses a different attention implementation than LLaMA/Gemma. Layer-by-layer tracing showed that hidden state differences grow linearly through Qwen's 28-layer stack:
  - Layer 1: diff ≈ 5e-7 (float32) / ≈ 1e-4 (float16)
  - Layer 10: diff ≈ 1e-5 (float32) / ≈ 1e-3 (float16)  
  - Layer 28: diff ≈ 2e-5 (float32) / ≈ 1e-2 (float16)
- In float32, the 2e-5 output diff only flips a token when the top-2 logits are exceptionally close.
- In float16, the 1e-2 output diff flips tokens **immediately** — often at the first or second generated token.

**Result**: On GPU, Qwen's outputs with KV reuse diverge from the reference within 1-2 tokens, producing semantically unrelated completions.

### Token-Level Divergence Pattern

```
Step  ref_token      reuse_token     Match
  0    come           come            ✓
  1    to             to              ✓  
  2    my             my              ✓
  3    house          house           ✓
  4    and            and             ✓
  5    we             we'll           �-  (first divergence)
  6    can            play            �-
  ...  (completely different after this point)
```

The first 4-5 tokens typically match (they're copied from the prompt suffix), but the first **generated** token often differs due to the accumulated float16 error in the hidden state.

### Why TinyLlama/Gemma/Phi-3 Don't Suffer

These architectures either:
- Use a **different attention normalization** (e.g., Gemma uses post-norm, LLaMA uses pre-norm) that bounds the error growth
- Have **fewer layers** or **different attention head counts** that change the error propagation
- Use **different position encoding** (e.g., RoPE with different base frequencies) that affects how the crop interacts with position embeddings

---

## 2. No-Reuse Control (ratio=0.0)

### What It Is

A control experiment where `shared_ratio = 0.0` — meaning **100% of the prompt is shuffled**, leaving zero shared prefix tokens between the original and modified prompts.

```python
shared = 0  # No shared tokens
cache.crop(0)  # Crop cache to empty
suffix = prompt_ids_mod[:, 0:]  # Prefill the ENTIRE modified prompt
```

With `shared = 0`, the cache crop removes everything. The suffix is the full modified prompt. Prefilling it on an **empty cache** is identical to a normal full prefill.

### Expected Result

**ROUGE-L = 1.0, 100% exact match**

Because no cache is actually reused — the operation degenerates to a clean generation from the modified prompt. Both `ref_text` and `reuse_text` are generated from the same prompt with the same `model.generate()` path (or the manual loop).

### Why Include It

1. **Validity check**: Confirms the pipeline doesn't produce false positives. If ratio=0.0 shows < 1.0, there's a bug in the code.
2. **Lower bound**: Demonstrates that when no reuse occurs, the output is identical.
3. **Paper narrative**: Shows that the engine only reuses when there's actual token overlap — for zero-overlap semantic matches, it falls through to a full prefill.

### Current Results (from local CPU tests)

| Model | Ratio=0.0 | Ratio=0.75 |
|-------|-----------|------------|
| TinyLlama | 1.0 | 1.0 |
| Qwen (CPU) | 1.0 | ~0.99 |

On CPU (float32), ratio=0.0 always shows 1.0. This confirms the pipeline is correct.

---

## 3. Float32 vs Float16 Comparison

### Why It Matters

The paper's experiments are typically run on GPU (float16) for speed, but the numerical behavior differs from CPU (float32). Comparing both precisions reveals whether the fidelity measurement is **precision-dependent**.

### Expected Differences

| Aspect | float32 (CPU) | float16 (GPU) |
|--------|--------------|--------------|
| Precision | ~7 decimal digits | ~3 decimal digits |
| KV cache error growth | ~2e-5 across 28 layers | ~1e-2 across 28 layers |
| Token divergence onset | Token 7+ | Token 1-2 |
| TinyLlama fidelity | 1.0 | 0.966 |
| Qwen fidelity | ~0.99 | 0.20 |

### What This Means for the Paper

1. **float16 amplifies architecture-dependent drift** — models that show minor drift on CPU (Qwen) become unusable on GPU
2. **LLaMA/Gemma are robust to precision change** — their fidelity drops from 1.0 to ~0.96-0.97, still excellent
3. **Recommendation**: Validate KV reuse fidelity in the deployment precision (typically float16 on GPU)
4. **Practical implication**: If using Qwen2-family models in float16, the `_partial_semantic_reuse` should use a conservative divergence threshold

### Recommended Experiment

Run TinyLlama at both precisions with identical prompts:

| Precision | Ratio | Exact Match | ROUGE-L | Notes |
|-----------|-------|-------------|---------|-------|
| float32 | 0.75 | 100% | 1.0 | CPU — already done |
| float16 | 0.75 | ~97% | ~0.97 | GPU — from Colab results |

The ~3% drop from float32 to float16 is the **precision cost** — acceptable for most applications but worth documenting.
