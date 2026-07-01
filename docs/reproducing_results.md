# Reproducing Results

This document describes how to reproduce the canonical HuggingFace and
real-world runtime experiment results.

## Canonical HF Results

The paper's main results use the HuggingFace backend on T4 and P100 GPUs.

### Setup

```bash
cd repo  # or the repository root
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
pip install pytest
```

### Run the test suite

```bash
python -m pytest -q
```

Expected: 49 passed, 1 skipped (the skipped test is an optional HF
KV-correctness check).

### Run a small reproduction

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

### Full paper reproduction

See the main `README.md` for the full 5-model, 10-dataset, 3-mode,
3-seed sweep script. Requires GPU access.

## Real-World Runtime Results

The runtime experiments (`runtime_experiments/`) use SGLang, LMCache,
and vLLM on RTX PRO 6000 Blackwell GPUs with Qwen2.5 models.

### Data format

Each `results.csv` file contains:

| Column | Description |
|--------|-------------|
| `model_slug` | Model identifier (e.g. `qwen25_7b`) |
| `model_params_B` | Parameter count in billions |
| `dataset` | Dataset name |
| `mode` | Prompt mode (`templated` or `rag`) |
| `engine` | Engine name |
| `mean_latency_ms` | Mean request latency in ms |
| `latency_ci_95_lower` | 95% CI lower bound |
| `latency_ci_95_upper` | 95% CI upper bound |
| `throughput_rps` | Throughput in requests/sec |
| `cached_tokens_mean` | Mean cached tokens per request |
| `gpu_energy_j` | Total GPU energy in Joules |
| `speedup_vs_lmcache_pct` | Speedup vs LMCache baseline (%) |

### Regenerate aggregate tables

```bash
cd runtime_experiments
python build_complete_tables.py
```

This reads raw data from the `v10/working/_extracted/` deliverables and
produces the clean CSV files in `sglang/`, `vllm/`, and `lmcache/`.
