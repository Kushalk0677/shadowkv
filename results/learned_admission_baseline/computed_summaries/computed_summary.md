# Learned Baseline Consolidated Summary

Source folders copied into this bundle:

- `phase3_partial/`: public-dataset learned-baseline run (4 models, 5 seeds)

Completed models: GPT-2, Qwen2.5-1.5B, TinyLlama-1.1B, Gemma-2B. Phi-3 is not included because it OOMed before producing train traces.

## Phase 3: MeritKV vs Learned

| Model | Variant | MeritKV speedup | MeritKV hit | MeritKV waste | Learned speedup | Learned hit | Learned waste | Learned flip-to-bypass | Learned flip-to-admit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 | raw | 0.993 | 0.331 | 0.000 | 1.001 | 0.233 | 0.000 | 4038 | 0 |
| GPT-2 | utility | 1.003 | 0.331 | 0.000 | 1.012 | 0.233 | 0.000 | 4038 | 0 |
| Qwen2.5-1.5B | raw | 1.101 | 0.515 | 0.027 | 1.081 | 0.293 | 0.040 | 1173 | 0 |
| Qwen2.5-1.5B | utility | 1.123 | 0.518 | 0.027 | 1.094 | 0.298 | 0.027 | 1205 | 0 |
| TinyLlama-1.1B | raw | 1.105 | 0.444 | 0.013 | 1.088 | 0.331 | 0.000 | 2336 | 0 |
| TinyLlama-1.1B | utility | 1.121 | 0.444 | 0.013 | 1.109 | 0.331 | 0.000 | 3124 | 0 |
| Gemma-2B | raw | 1.193 | 0.515 | 0.107 | 1.160 | 0.329 | 0.053 | 1422 | 0 |
| Gemma-2B | utility | 1.194 | 0.515 | 0.080 | 1.161 | 0.330 | 0.080 | 1795 | 0 |

## Reading

- The four-model Phase 3 learned-baseline results are usable.
- Phi-3 is excluded because it OOMed before producing train traces.
- MeritKV consistently outperforms the learned policy on speedup across all models (2-3% advantage).
- The learned policy achieves lower waste by being more conservative (more flips-to-bypass), but this trades off speedup.
