# Canonical Result Bundle
This directory contains the two main 3-seed hardware result sets used for the ShadowKV++ draft.
## Included Results
- Benchmark JSON files: `898`
- Engine rows: `8532`
- Hardware result roots: `results/final_p100`, `results/final_t4`
- Seeds: `42`, `123`, `456`
- Models: GPT-2, TinyLlama-1.1B-Chat, Qwen2.5-1.5B-Instruct, Gemma-2B-IT, Phi-3-mini-4k-instruct
- Datasets: AG News, Banking77, AlpacaEval, Dolly, DailyDialog, OASST1, UltraChat, SAMSum, XSum, CNN/DailyMail

## Headline Aggregate
| Engine | Mean Speedup | 95% CI | P95 Speedup | Waste | Hit Rate |
|---|---:|---:|---:|---:|---:|
| `frequency_speculative` | 1.208x | [1.191, 1.224] | 1.209x | 0.284 | 0.617 |
| `greedy_prefix_cache` | 1.221x | [1.203, 1.238] | 1.184x | 0.000 | 0.320 |
| `no_cache` | 1.000x | [1.000, 1.000] | 1.000x | 0.000 | 0.000 |
| `reactive_prefix_cache` | 1.214x | [1.198, 1.229] | 1.134x | 0.000 | 0.317 |
| `shadow_kv` | 1.287x | [1.268, 1.306] | 1.318x | 0.264 | 0.606 |
| `shadow_kv_plus` | 1.365x | [1.342, 1.388] | 1.541x | 0.156 | 0.402 |
| `shadow_kv_plus_best_latency` | 1.331x | [1.307, 1.354] | 1.455x | 0.228 | 0.500 |
| `shadow_kv_plus_raw_observer` | 1.356x | [1.333, 1.379] | 1.514x | 0.158 | 0.404 |
| `strict_reactive_prefix_cache` | 1.254x | [1.236, 1.273] | 1.278x | 0.000 | 0.310 |

## Notes
- Raw-mode ShadowKV++ gains should be interpreted as bypass/overhead-avoidance gains, not as exact KV reuse gains.
- Approximate semantic KV substitution is not correctness-preserving by default; semantic opportunities should be interpreted with the paper's correctness boundary.
- `summary_by_engine.csv` and `summary_by_mode_engine.csv` are generated from the JSON files in this folder.
