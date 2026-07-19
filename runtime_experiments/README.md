# Runtime Experiments

This directory is organized by model family. Each model-family folder owns its curated runtime tables, summaries, and local raw run files.

## Layout

| Path | Contents |
|---|---|
| `qwen2.5/` | Qwen2.5 runtime results, curated tables, summaries, k-star outputs, and included run files. |
| `gemma4/` | Measured Gemma 4 Blackwell runtime results across no-cache, native runtime-cache, and MeritKV arms. |

## Notes

- Gemma 4 is populated with measured 5-seed Blackwell runtime results for vLLM APC, SGLang RadixAttention, and LMCache + vLLM, each with no-cache and MeritKV arms.
- Qwen2.5 remains organized with `vllm/`, `sglang/`, `lmcache/`, and `kstar/` outputs.
- Raw result trees may be kept locally and ignored by Git when they are too large for the public tracked tree.
