# Semantic Fidelity and Correctness Boundary

This document describes how the repository treats semantic KV reuse. The short version is simple: semantic matching is useful, but approximate semantic KV substitution is not assumed to be correctness-preserving by default.

## Claim Boundary

ShadowKV++ makes two separate claims:

1. It can identify and score semantic reuse opportunities that exact-prefix caches miss.
2. It can explicitly bypass or block approximate reuse when the request-level utility or correctness boundary is not good enough.

That is different from claiming that arbitrary semantic KV reuse is safe. The default `shadow_kv_plus` path on real HF-style backends is conservative: it records semantic opportunity metrics and avoids unsafe approximate KV substitution unless an explicit ablation or guarded execution path is enabled.

## Fidelity Question

The fidelity experiment asks:

> If cached KV state from one prompt prefix is reused inside another prompt computation, does the generated output change?

The test uses `DynamicCache.crop()` to keep a shared prefix and then compares clean generation against reuse-assisted generation.

```text
prompt A -> prefill -> crop shared prefix -> continue prompt B suffix -> generate reuse_text
prompt B -> clean generation -----------------------------------------> generate ref_text
```

The key comparison is `ref_text` versus `reuse_text`.

## Why This Matters

Prefix-identical reuse is expected to preserve semantics when the backend implements it correctly. Approximate semantic reuse is different: two requests may mean similar things without sharing the same token history. Reusing KV tensors across that boundary can change hidden states, logits, and generated text.

For that reason, the paper should describe semantic reuse as a correctness-bounded opportunity, not as a blanket production-safe optimization.

## Example Results

The public example files live in:

```text
results/fidelity_examples/
  f16/
  f32/
```

The float16 examples show model-dependent behavior. TinyLlama and Gemma-style models were comparatively robust in these checks, while Qwen-style models showed much larger sensitivity. These examples should be used as diagnostic evidence, not as universal guarantees.

## Interpretation

Use this language in the paper and README:

- ShadowKV++ detects semantic opportunities that exact-prefix caching misses.
- The controller scores whether reuse is useful for the current request.
- The system can bypass low-utility reuse.
- On real backends, unsafe approximate semantic KV reuse is blocked by default unless a guarded or ablation path is explicitly enabled.
- Fidelity must be validated per model family and deployment precision.

Avoid this language:

- Semantic KV reuse is always safe.
- Approximate semantic reuse maintains output quality in general.
- Semantic hit rate alone proves performance improvement.

## Related Files

| File | Purpose |
|------|---------|
| `experiments/run_fidelity_equiv.py` | Fidelity experiment pipeline |
| `experiments/eval_comprehensive.py` | ROUGE-L and exact-match evaluation |
| `results/fidelity_examples/README.md` | Public example-data guide |
| `docs/fidelity_deep_analysis.md` | Qwen float16 sensitivity analysis |
| `docs/semantic_correctness_ablations.md` | Scaffold-only, early-layer, and logit-guard ablation notes |
| `docs/semantic_paraphrase_novelty.md` | Semantic workload and novelty explanation |

## Reproduction Sketch

Small CPU check:

```bash
python experiments/run_fidelity_equiv.py --device cpu \
  --models tinyllama \
  --datasets samsum alpaca_eval banking77 \
  --n_samples 8 --max_gen_tokens 64
```

GPU checks should be interpreted with the actual deployment dtype, typically float16 or bfloat16, because precision can change the observed fidelity.
