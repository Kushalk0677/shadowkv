# Runtime Experiments

This directory is organized by model family. Each model-family folder owns its runtime results and raw run files.

## Layout

| Path | Contents |
|---|---|
| `qwen2.5/` | Current Qwen2.5 runtime results, curated tables, summaries, and included run files. |
| `gemma4/` | Gemma 4 Blackwell runtime baseline matrix, curated tables, summaries, and included run files. |

## Notes

- Qwen2.5 is the complete populated family at the moment.
- Gemma 4 is populated with the Blackwell runtime baseline matrix for vLLM APC, SGLang RadixAttention, and LMCache + vLLM.
- Runtime-specific organization is preserved inside each model-family folder: `vllm/`, `sglang/`, `lmcache/`, and `kstar/`.

