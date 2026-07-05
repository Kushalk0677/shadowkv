# KV Cache Reuse Fidelity: Example Data

This directory contains per-sample fidelity experiment data for the evaluation models. Each JSON file stores generation-level outputs that can be used to compare reference text against reuse text.

## Structure

```text
fidelity_examples/
  README.md
  f16/                     # Float16 GPU examples
    all_results.json
    tinyllama_results.json
    gpt2_results.json
    qwen25_15b_results.json
    gemma2b_results.json
    phi3mini_results.json
  f32/                     # Float32 CPU control examples
    gpt2_results.json
```

## Float16 Examples

| Model | File |
|---|---|
| GPT-2 | `f16/gpt2_results.json` |
| TinyLlama | `f16/tinyllama_results.json` |
| Qwen2.5 1.5B | `f16/qwen25_15b_results.json` |
| Gemma 2B | `f16/gemma2b_results.json` |
| Phi-3 Mini | `f16/phi3mini_results.json` |
| Combined index/summary | `f16/all_results.json` |

## Float32 Control

The current float32 control file is:

```text
f32/gpt2_results.json
```

Use this folder as example data, not as the primary aggregate result table. The primary performance summaries live in `results/controlled_results/summary_by_engine.csv` and `results/controlled_results/summary_by_mode_engine.csv`.

## Format

Each per-sample JSON entry may include fields such as:

```text
model
dataset
shared_ratio
exact_text
ref_text
reuse_text
shared_tokens
total_tokens_orig
```

Fidelity is computed by comparing `ref_text` and `reuse_text`, commonly with ROUGE-L or a similar text-overlap metric.

## Interpretation Notes

- Approximate semantic KV reuse is not automatically correctness-preserving.
- Treat these examples as diagnostic evidence for when reuse is safe, risky, or model-dependent.
- Qwen-style models have shown sensitivity in prior checks, so semantic reuse claims should be guarded rather than absolute.
