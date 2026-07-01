# ShadowKV++: Waste-Aware, Correctness-Bounded KV Cache Reuse

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](PENDING)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ShadowKV++ is a per-request policy controller for KV cache reuse in LLM serving. It scores each potential reuse event with a net-utility objective $U = B - C - W$ (benefit, cost, waste) and admits reuse only when $U \geq 0$. Three properties distinguish it: (1) bypass is a first-class action for low-signal workloads; (2) semantic neighbourhood detection identifies candidate reuse opportunities, which are admitted only when $U \geq 0$ and a backend execution flag is set; and (3) per-request telemetry records every policy decision for audit.

> **Authors:** Kushal Khemani, Evan Leri, Dr. Sparsh Mittal

---

## How It Works

ShadowKV++ combines three cooperating components:

- **TieredStateBank** — stores KV entries keyed by prefix token sequences with longest-prefix lookup via a radix trie. Per-prefix statistics track frequency, observation count, branching factor, and memory footprint. Entries are promoted to GPU-resident tier after sufficient reuse events and demoted under memory pressure.
- **AdaptiveReuseController** — calls `Plan(x)` per request and returns a `ReusePlan`: strategy, speculate depth, reusable prefix length, expected benefit/cost/waste (ms), score, confidence, and reason. Three branches handle exact-prefix reuse, semantic-approximate reuse (when $U_s \geq 0$ and the execution flag is set), and bypass when utility is negative.
- **SemanticKVIndex** — detects paraphrase-equivalent request families with a 128-dim hash-based token sketch without a sentence encoder on the hot path. Cosine similarity between normalised vectors approximates structural overlap.

---

## Key Results

### Main HF Evaluation (5 models �- 10 datasets �- 5 seeds)

| Engine | Mean Speedup | Waste | Hit Rate |
|--------|:-----------:|:-----:|:--------:|
| No cache | 1.000�- | 0.000 | 0.000 |
| Reactive | 1.214�- | 0.000 | 0.317 |
| Greedy | 1.221�- | 0.000 | 0.320 |
| Strict reactive | 1.254�- | 0.000 | 0.310 |
| Frequency spec. | 1.208�- | 0.284 | 0.617 |
| ShadowKV | 1.287�- | 0.264 | 0.606 |
| **ShadowKV++** | **1.365�-** | **0.156** | **0.402** |

- Waste 41% below ShadowKV (0.156 vs 0.264)
- Significant on all 10 datasets ($p < 0.001$, nine datasets; $p < 10^{-4}$, one dataset)

### Runtime Evaluation (SGLang, LMCache, vLLM — Qwen2.5 1.5B–32B)

| Metric | Value |
|--------|-------|
| ShadowKV++ vs LMCache at 7B | +16.7% |
| ShadowKV++ at 32B over LMCache | +5.1% |
| vLLM APC+ShadowKV++ vs no-cache at 32B | +19.0% |
| GPU energy saving (vLLM at 32B) | ≈25% |

### KV Cache Reuse Fidelity (DynamicCache.crop(), 75% shared prefix, float16)

| Model | ROUGE-L | Verdict |
|-------|:-------:|:-------:|
| TinyLlama (1.1B) | **0.966** | Safe |
| Gemma 2B | **0.974** | Safe |
| Phi-3 Mini (3.8B) | **0.931** | Acceptable |
| GPT-2 (124M) | **0.876** | Acceptable |
| Qwen2.5 1.5B | **0.200** | Needs precision guard |

### Controller Overhead (Plan() latency, 1,000 synthetic requests)

| Component | Mean | Frequency |
|-----------|:---:|:---------:|
| Policy planning | 2.30 ms | 100% |
| Semantic index query | 7.18 ms | 17% |
| Amortised overhead | 1.2 ms | — |
| **Fraction of GPU inference time** | **0.5–2.3%** | — |

---

## Repository Layout

```
src/proactive_kv_cache/        Core engines, cache bank, controller, models, policy
├── engines.py                 BaseEngine, NoCacheEngine, ShadowKVPlusEngine
├── cache.py                   TieredStateBank with radix trie
├── controller.py              AdaptiveReuseController, utility scoring
├── policy.py                  CostAwareSlackPolicy
├── semantic.py                SemanticKVIndex, token sketching
├── models.py                  Backend abstraction (FakeBackend, HFBackend)
├── datasets.py                Dataset loading and prompt templates
├── utils.py                   Calibration, estimates, KV byte counting
├── metrics.py                 Engine metrics, aggregation
├── policy_learning.py         Offline grid-search learner
└── workload.py                Workload generation
├── experiments/
│   ├── run_benchmark.py       Main benchmark entry point
│   ├── run_fidelity_equiv.py  KV cache reuse fidelity pipeline
│   ├── eval_comprehensive.py  ROUGE-L and exact-match evaluator
│   ├── profile_plan.py        Controller Plan() latency profiler
│   ├── analyze_shadowkv_results.py
│   ├── archive/               Superseded experiment scripts
│   └── ...
├── results/
│   ├── final_p100/            Canonical P100 benchmark JSONs
│   ├── final_t4/              Canonical T4 benchmark JSONs
│   ├── fidelity/              Fidelity experiment JSONs (5 models �- 10 datasets)
│   │   ├── all_results.json   1,221 samples
│   │   ├── control/           Ratio=0.0 control (13 samples, 100% match)
│   │   └── qwen_ratios/       Multi-ratio sweep (75/50/25%)
│   ├── RESULTS.md             Headline aggregate table
│   └── summary_by_engine.csv
├── docs/
│   ├── semantic_fidelity.md   Full fidelity report
│   ├── fidelity_deep_analysis.md  Root cause analysis
│   ├── experimental_setup.md  Methodology description
│   ├── results_table.md       Complete results with precision comparison
│   ├── reports/               Archived design reports
│   └── ...
├── runtime_experiments/       SGLang, LMCache, vLLM results
├── tests/                     Unit and regression tests
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .\.venv\Scripts\Activate.ps1       # Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
pip install pytest
```

### 3. Verify with a smoke test (FakeBackend, no GPU required)

```bash
python experiments/run_benchmark.py \
  --backend fake \
  --workload synthetic \
  --variant high_skew \
  --n_requests 40 \
  --include_experimental \
  --disable_arrival_simulation \
  --output_dir results/smoke_fake
```

Expected output: benchmark JSON and summary in `results/smoke_fake/`.

### 4. Run the test suite

```bash
python -m pytest -q
```

Expected: `49 passed, 1 skipped`. The skipped test is an optional slow HF KV-correctness check.

---

## Using the Controller

```python
from proactive_kv_cache.engines import ShadowKVPlusEngine
from proactive_kv_cache.models import FakeBackend

backend = FakeBackend()
engine = ShadowKVPlusEngine(backend=backend, max_memory_mb=256)

tokens = tuple(range(50, 100))  # synthetic token sequence
metadata = {'prompt_mode': 'raw', 'arrival_time': 0.0}

result = engine.serve_tokens(request_id=0, tokens=tokens, metadata=metadata)
print(f"Reused {result.matched_prefix_length} tokens, "
      f"recomputed {result.tokens_recomputed}, "
      f"latency {result.latency_ms:.2f} ms")
```

---

## Experiments

| Experiment | Description |
|---|---|
| `run_benchmark.py` | Full HF benchmark: 5 models, 10 datasets, 3 modes, 5 seeds |
| `run_fidelity_equiv.py` | KV reuse fidelity via DynamicCache.crop() |
| `profile_plan.py` | Controller Plan() latency profiling |
| `eval_comprehensive.py` | ROUGE-L and exact-match evaluation |
| `analyze_shadowkv_results.py` | Result parser and policy-summary generator |

### Reproduce a small matrix

```bash
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="ag_news"
N=64
OUT="results/reproduction_qwen_agnews"

for SEED in 42 123 456; do
  for MODE in raw templated semantic; do
    python experiments/run_benchmark.py \
      --backend hf --model "$MODEL" --device cuda:0 --dtype float16 \
      --workload public_dataset --dataset "$DATASET" --prompt_mode "$MODE" \
      --n_requests "$N" --seed "$SEED" --include_experimental \
      --disable_arrival_simulation --output_dir "$OUT/$MODE/seed_$SEED"
  done
done
```

### Reproduce fidelity results (CPU)

```bash
python experiments/run_fidelity_equiv.py --device cpu \
  --models tinyllama \
  --datasets samsum alpaca_eval banking77 \
  --n_samples 8 --max_gen_tokens 64
```

### GPU fidelity (Colab T4)

Upload [`experiments/fidelity_equiv_colab.ipynb`](experiments/fidelity_equiv_colab.ipynb) to Google Colab and run.

### Controller overhead profiling

```bash
python experiments/profile_plan.py
```

---

## Included Result Artifacts

| Device / Experiment | Path |
|---|---|
| T4 benchmark JSONs (3 seeds) | `results/final_t4/` |
| P100 benchmark JSONs (3 seeds) | `results/final_p100/` |
| Fidelity experiment (5 models, ~1,200 samples) | `results/fidelity/` |
| Fidelity control (ratio=0.0) | `results/fidelity/control/` |
| Qwen multi-ratio sweep | `results/fidelity/qwen_ratios/` |
| Runtime experiments (SGLang, LMCache, vLLM) | `runtime_experiments/` |

---

## What Is Not Included

To keep the repository lightweight and GitHub-friendly:

- **Paper source / LaTeX** — the manuscript is maintained separately
- **Model weights** — all models are downloaded from HuggingFace at runtime
- **Datasets** — all datasets are downloaded from HuggingFace at runtime
- **Transient local outputs** outside the curated `results/` and `runtime_experiments/` snapshots
- **GPU benchmark baselines** — full reproduction requires a CUDA-capable GPU (T4, P100, or better)

---

## Citation

If you use ShadowKV++ in your research, please cite the arXiv paper:

```bibtex
@misc{khemani2025shadowkv,
  title={ShadowKV++: A Policy Controller for Waste-Aware, Admission-Controlled KV Cache Reuse in LLM Serving},
  author={Kushal Khemani and Evan Leri and Sparsh Mittal},
  year={2025},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/XXXX.XXXXX},
}
```

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.
