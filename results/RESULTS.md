# Canonical Result Bundle

## Engine Name Aliases

Raw artifacts keep the stable engine IDs used during execution. Public-facing text maps them as follows:

| Engine ID | Display name |
|---|---|
| `shadow_kv_plus` | MeritKV |
| `shadow_kv` | MeritKV-Sem |
| `shadow_kv_plus_lite` | MeritKV-Lite |


This directory is the public result bundle for the MeritKV draft. It reports `shadow_kv` as MeritKV-Sem, `shadow_kv_plus` as MeritKV, and `shadow_kv_plus_lite` as MeritKV-Lite. It keeps the controlled aggregate results, realistic isolated-result traces, fidelity examples, and small timing smoke outputs in one place.

## Directory Layout

```text
results/
  controlled_results/     # T4/P100 controlled multi-engine benchmark outputs and CSV summaries
  realistic_results/      # Process-isolated no_cache and MeritKV (`shadow_kv_plus`) JSON outputs
  fidelity_examples/      # Per-sample KV reuse fidelity examples
  sweep_timing/           # Small fake-backend timing smoke outputs
  RESULTS.md              # This overview
  architectural_robustness.md
```

## Controlled Results

`controlled_results/` contains the aggregate tables used for the main paper-style comparison.

Included files:

```text
controlled_results/summary_by_engine.csv
controlled_results/summary_by_mode_engine.csv
controlled_results/manifest.json
controlled_results/t4/**/benchmark_*.json
controlled_results/p100/**/benchmark_*.json
```

Current controlled bundle contents:

```text
Benchmark JSON files: 898 (900 planned; 2 Phi-3 templated samsum runs on T4 were unavailable in the source bundle. See manifest.json for details.)
Engine rows summarized: 8532
Hardware roots: controlled_results/t4, controlled_results/p100
Seeds: 42, 123, 456
Prompt modes: raw, templated, semantic
Models: GPT-2, TinyLlama-1.1B-Chat, Qwen2.5-1.5B-Instruct, Gemma-2B-IT, Phi-3-mini-4k-instruct
Datasets: AG News, Banking77, AlpacaEval, Dolly, DailyDialog, OASST1, UltraChat, SAMSum, XSum, CNN/DailyMail
```

### Headline Controlled Aggregate

| Engine | Mean Speedup | 95% CI | P95 Speedup | Waste | Hit Rate |
|---|---:|---:|---:|---:|---:|
| `frequency_speculative` | 1.208x | [1.191, 1.224] | 1.209x | 0.284 | 0.617 |
| `greedy_prefix_cache` | 1.221x | [1.203, 1.238] | 1.184x | 0.000 | 0.320 |
| `no_cache` | 1.000x | [1.000, 1.000] | 1.000x | 0.000 | 0.000 |
| `reactive_prefix_cache` | 1.214x | [1.198, 1.229] | 1.134x | 0.000 | 0.317 |
| MeritKV-Sem (`shadow_kv`) | 1.287x | [1.268, 1.306] | 1.318x | 0.264 | 0.606 |
| MeritKV (`shadow_kv_plus`) | 1.365x | [1.342, 1.388] | 1.541x | 0.156 | 0.402 |
| MeritKV-BestLatency (`shadow_kv_plus_best_latency`) | 1.331x | [1.307, 1.354] | 1.455x | 0.228 | 0.500 |
| MeritKV-RawObserver (`shadow_kv_plus_raw_observer`) | 1.356x | [1.333, 1.379] | 1.514x | 0.158 | 0.404 |
| `strict_reactive_prefix_cache` | 1.254x | [1.236, 1.273] | 1.278x | 0.000 | 0.310 |

The CSV files contain the authoritative aggregate values. Prefer the CSVs over copying table values by hand.

## Realistic Results

`realistic_results/` contains process-isolated JSON outputs for deployment-style checks. The current folder has 3000 JSON files and one sweep log. It is organized by engine first:

```text
realistic_results/no_cache/<model>/<prompt_mode>/seed_<seed>/<dataset>/benchmark_*.json
realistic_results/shadow_kv_plus/<model>/<prompt_mode>/seed_<seed>/<dataset>/benchmark_*.json  # MeritKV
```

Use these files to inspect isolated no-cache and MeritKV behavior under a cleaner per-engine process boundary. These are not the same as the controlled aggregate CSVs.

## Fidelity Examples

`fidelity_examples/` contains per-sample generation outputs for KV reuse fidelity checks. See `fidelity_examples/README.md` for format and caveats.

## Interpretation Notes

- Raw-mode MeritKV gains should be interpreted as bypass and overhead-avoidance gains, not as proof of exact KV reuse.
- Approximate semantic KV substitution is not correctness-preserving by default. Semantic opportunities should be discussed with explicit correctness boundaries.
- High hit rate alone is not a win. Use matched latency and energy columns where available.
- `controlled_results/summary_by_engine.csv` and `controlled_results/summary_by_mode_engine.csv` are generated from the controlled benchmark JSON files.

