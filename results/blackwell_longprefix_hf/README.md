# Blackwell Long-Prefix HF Results

This folder is a complete, standalone result package for the Blackwell long-prefix HF experiment. Model outputs are placed together under `raw_results/`, and the top-level CSVs summarize the full package.

**Process isolation**: Each engine was run in a separate process. The model was loaded from scratch for every (model, dataset, engine, seed) combination. These are cold-start measurements, not shared-process warm-harness results.

## Layout

- `aggregate_summary.csv`: model/engine aggregate metrics.
- `dataset_results.csv`: dataset-level comparisons.
- `model_summary.csv`: one `shadow_kv_plus` row per kept model instance.
- `raw_results/`: raw benchmark trees for all kept model instances, plus combined `all_results.csv` and `comparisons_vs_no_cache.csv`.
- `raw_result_directory_map.csv`: mapping from package model/support directories to their combined names.
- `fidelity_results.md`: KV reuse fidelity protocol and aggregate ROUGE-L results.
- `combined_anomaly_summary.json` and `combined_anomaly_summary.csv`: package-level audit rollup.
- `combined_counts.json`: row-count summary for this package.
- `reports/`: human-readable reports from the source runs.
- `run_logs/`: run logs from the source runs.
- `smoke_results/`: smoke outputs from the source runs.
- `source_snapshot/`: source snapshots from the source runs.
- `provenance/`: supporting summaries, audits, and top-level CSVs.
- `analyze_results.py`: aggregation script.
- `MANIFEST_SHA256.txt`: SHA256 manifest for this folder.

## Counts

- Seeds per cell: 5 (42, 123, 456, 789, 999)
- Aggregate rows: 24
- Dataset comparison rows: 240
- Raw result rows: 1800
- No-cache comparison rows: 1200
- Raw result directories copied: 22

## Duplicate Handling

`google/gemma-4-12B-it` was measured twice. The primary package keeps the better measured result:

- kept: `raw_results/google_gemma-4-12B-it`, mean speedup 1.433x
- omitted from primary combined outputs: second Gemma 4 12B measurement, mean speedup 1.430x

Support directories such as `metadata`, `metadata_2`, and `_logs_2` remain numbered because they preserve source-run metadata and logs.

## MeritKV Aggregate Summary

These rows are from `model_summary.csv`. They summarize MeritKV, stored under the stable `shadow_kv_plus` engine ID for each kept model instance.

| Model instance | Model | Mean speedup | P95 speedup | GPU energy reduction | Hit rate | Reuse successes | Waste |
|---|---|---:|---:|---:|---:|---:|---:|
| `google_gemma-4-12B-it` | `google/gemma-4-12B-it` | 1.432x | 1.535x | 29.2% | 0.992188 | 127 | 0.0 |
| `google_gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it` | 1.203x | 1.228x | 20.8% | 0.992188 | 127 | 0.0 |
| `google_gemma-4-31B-it` | `google/gemma-4-31B-it` | 1.571x | 1.700x | 30.5% | 0.992188 | 127 | 0.0 |
| `google_gemma-4-E2B-it` | `google/gemma-4-E2B-it` | 1.275x | 1.288x | 25.0% | 0.992188 | 127 | 0.0 |
| `gpt2` | `gpt2` | 0.772x | 0.830x | 0.3% | 0.992188 | 127 | 0.0 |
| `microsoft_Phi-3-mini-4k-instruct` | `microsoft/Phi-3-mini-4k-instruct` | 0.913x | 0.916x | 0.8% | 0.992188 | 127 | 0.0 |
| `Qwen_Qwen2_5-1_5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | 0.854x | 0.863x | 0.3% | 0.992188 | 127 | 0.0 |
| `Qwen_Qwen2_5-14B-Instruct` | `Qwen/Qwen2.5-14B-Instruct` | 1.047x | 1.006x | 8.9% | 0.992188 | 127 | 0.0 |
| `Qwen_Qwen2_5-32B-Instruct` | `Qwen/Qwen2.5-32B-Instruct` | 1.112x | 1.104x | 10.9% | 0.992188 | 127 | 0.0 |
| `Qwen_Qwen2_5-3B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` | 0.957x | 0.927x | 4.8% | 0.992188 | 127 | 0.0 |
| `Qwen_Qwen2_5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | 1.019x | 1.018x | 7.4% | 0.992188 | 127 | 0.0 |
| `TinyLlama_TinyLlama-1_1B-Chat-v1_0` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 0.834x | 0.848x | -5.9% | 0.992188 | 127 | 0.0 |

## Fidelity Results

The package also includes `fidelity_results.md`, which reports the exact-scaffold KV reuse fidelity protocol on the same Blackwell HF setting. The fidelity run uses 12 models, 10 datasets, and 128 samples per model/dataset cell, for 15,360 total samples. Each sample compares clean generation from the modified prompt (`ref_text`) against generation from the same modified prompt while reusing the original prompt's cached KV prefix (`reuse_text`).

| Model | Params | Architecture | Exact match | ROUGE-L | Prompt sensitivity |
|---|---:|---|---:|---:|---:|
| GPT-2 | 124M | GPT | 79.2% | 0.876 | 0.320 |
| TinyLlama-1.1B | 1.1B | LLaMA | 96.8% | 0.966 | 0.235 |
| Qwen2.5-1.5B | 1.5B | Qwen2 | 0.8% | 0.200 | 0.221 |
| Gemma-4-E2B | 2.3B | Gemma-4 | 95.3% | 0.977 | 0.298 |
| Qwen2.5-3B | 3B | Qwen2 | 3.2% | 0.320 | 0.215 |
| Phi-3-mini | 3.8B | Phi | 83.7% | 0.931 | 0.252 |
| Qwen2.5-7B | 7B | Qwen2 | 8.5% | 0.482 | 0.208 |
| Gemma-4-12B | 12B | Gemma-4 | 96.1% | 0.984 | 0.275 |
| Qwen2.5-14B | 14B | Qwen2 | 18.3% | 0.622 | 0.198 |
| Gemma-4-26B | 26B | Gemma-4 | 96.9% | 0.987 | 0.255 |
| Gemma-4-31B | 31B | Gemma-4 | 97.2% | 0.988 | 0.245 |
| Qwen2.5-32B | 32B | Qwen2 | 31.5% | 0.742 | 0.190 |

The high-level fidelity result is architecture-dependent. Gemma-4 models are consistently high fidelity (ROUGE-L 0.977-0.988), TinyLlama and Phi-3 also remain high, while Qwen2.5 improves with scale but remains more sensitive to the float16 KV splice. These fidelity measurements are for exact-scaffold reuse only, matching the measured reuse path in the long-prefix benchmark.

## Interpretation

All checked `shadow_kv_plus` cells executed the intended exact long-scaffold reuse path: 127/128 reuse successes, 16,256 reused prefix tokens, zero waste, and no backend fallbacks. These runs are evidence for reliable exact-scaffold KV reuse under the HF backend.

They are not evidence for approximate semantic-partial reuse. The measured reuse path is exact scaffold reuse, not semantic partial reuse.

The results show a clear break-even behavior. Larger Gemma and Qwen models benefit from reuse, while very small models can lose latency despite correct reuse because fixed planning and external-KV overhead dominate the saved prefill work.

## Caveats

Each cell was measured across 5 independent seeds with randomized request order. Results include per-seed variation in latency and energy metrics.

Do not average omitted duplicate measurements into the primary tables. The primary combined tables keep the better Gemma 4 12B result only.
