# Reproducing Results

Raw artifacts keep stable engine IDs: `shadow_kv_plus` displays as MeritKV, `shadow_kv` displays as MeritKV-Sem, and `shadow_kv_plus_lite` displays as MeritKV-Lite.


This document describes how to reproduce the repository's HuggingFace benchmark checks and how to read the runtime experiment tables.

## Canonical HF Results

The main controlled results use the HuggingFace backend on T4 and P100 GPUs. The public result bundle is stored under `results/controlled_results/`.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
pip install pytest
```

### Run the Test Suite

```bash
python -m pytest -q
```

The exact count can change as tests are added. A slow HF KV-correctness test may be skipped when the required model/GPU environment is not available.

### Run a Small Reproduction

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model distilgpt2 \
  --device cpu \
  --workload public_dataset \
  --dataset ag_news \
  --prompt_mode templated \
  --n_requests 32 \
  --include_experimental \
  --disable_arrival_simulation \
  --output_dir results/hf_cpu_agnews_templated
```

### Full Paper-Style Sweep

The public aggregate results summarize 5 models, 10 datasets, 3 prompt modes, and 3 seeds (`42`, `123`, `456`) for T4 and P100 controlled runs. See the main `README.md` for a smaller example loop. A full sweep requires CUDA GPU access and enough time to run all model/dataset/mode combinations.

## Result Layout

```text
results/
  controlled_results/     # T4/P100 controlled JSONs and aggregate CSVs
  realistic_results/      # Process-isolated no_cache and MeritKV (`shadow_kv_plus`) JSONs
  fidelity_examples/      # Per-sample fidelity examples
  sweep_timing/           # Small timing/smoke outputs
```

Primary aggregate files:

```text
results/controlled_results/summary_by_engine.csv
results/controlled_results/summary_by_mode_engine.csv
results/controlled_results/manifest.json
```

## Runtime Results

The runtime experiments in `runtime_experiments/` use SGLang, LMCache, and vLLM on an RTX PRO 6000 Blackwell GPU with Qwen2.5-family models.

Each runtime CSV table contains columns such as:

| Column | Description |
|--------|-------------|
| `model_slug` | Model identifier |
| `model_params_B` | Parameter count in billions |
| `dataset` | Dataset name |
| `mode` | Prompt mode |
| `engine` | Engine name |
| `mean_latency_ms` | Mean request latency in ms |
| `latency_ci_95_lower` | 95% CI lower bound, when available |
| `latency_ci_95_upper` | 95% CI upper bound, when available |
| `throughput_rps` | Throughput in requests/sec |
| `cached_tokens_mean` | Mean cached tokens per request |
| `gpu_energy_j` | Total GPU energy in Joules |
| `speedup_vs_lmcache_pct` | Speedup vs LMCache baseline, where applicable |

The compact CSVs in this public repo are the curated tables. Full raw runtime deliverables are not included here; regenerate them only if you have the external runtime systems and original measurement environment.

### Regenerate Aggregate Tables

```bash
cd runtime_experiments
python build_complete_tables.py
```

This command is intended for a working copy that also has the raw runtime deliverables available. In the public repo, treat the checked-in CSVs as the compact release artifacts.
