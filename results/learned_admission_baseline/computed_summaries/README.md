# Learned Baseline Consolidated Summary

Source folders copied into this bundle:

- `phase3_partial/`: public-dataset learned-baseline run, stopped before Phi-3 because Phi-3 OOMed at model load.
- `memory_bound_fixed/`: corrected memory-bound rerun for the four completed models.

Completed models: GPT-2, Qwen2.5-1.5B, TinyLlama-1.1B, Gemma-2B. Phi-3 is not included in the computed summaries.

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

## Fixed Memory-Bound Trace

Evictions are computed from per-request `evictions_cumulative` in `memory_bound_trace.json`, not from the aggregate JSON field in the downloaded run.

| Model | Engine | Mean evictions/seed | Warmup hit | Pressure hit | Recovery hit | Recovery speedup | Learned admit | Learned bypass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 | shadow_kv_plus | 47.0 | 0.500 | 0.500 | 0.000 | 1.000 | 0 | 0 |
| GPT-2 | learned_raw | 0.0 | 0.000 | 0.000 | 0.000 | 1.003 | 425 | 15 |
| GPT-2 | learned_utility | 0.0 | 0.000 | 0.000 | 0.000 | 1.001 | 425 | 15 |
| Qwen2.5-1.5B | shadow_kv_plus | 49.0 | 0.500 | 0.500 | 0.000 | 0.999 | 0 | 0 |
| Qwen2.5-1.5B | learned_raw | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 440 |
| Qwen2.5-1.5B | learned_utility | 0.0 | 0.000 | 0.000 | 0.000 | 0.998 | 0 | 440 |
| TinyLlama-1.1B | shadow_kv_plus | 49.0 | 0.500 | 0.500 | 0.000 | 0.999 | 0 | 0 |
| TinyLlama-1.1B | learned_raw | 49.0 | 0.500 | 0.500 | 0.000 | 0.999 | 190 | 250 |
| TinyLlama-1.1B | learned_utility | 49.0 | 0.500 | 0.500 | 0.000 | 0.999 | 190 | 250 |
| Gemma-2B | shadow_kv_plus | 49.0 | 0.500 | 0.500 | 0.000 | 0.999 | 0 | 0 |
| Gemma-2B | learned_raw | 49.0 | 0.500 | 0.500 | 0.000 | 1.000 | 190 | 250 |
| Gemma-2B | learned_utility | 49.0 | 0.500 | 0.500 | 0.000 | 1.001 | 190 | 250 |

## Reading

- The four-model Phase 3 learned-baseline results are usable.
- Phi-3 is excluded because it OOMed before producing train traces.
- The corrected memory trace creates eviction pressure for MeritKV-style engines, but victim recovery remains zero in this synthetic trace.
- Treat the memory-bound trace as a negative/neutral diagnostic, not as positive support for locality preservation.
