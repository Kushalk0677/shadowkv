# ShadowKV++

ShadowKV++ is a research prototype for **waste-aware, correctness-bounded KV cache reuse** in LLM serving. It extends a tiered prefix-cache benchmark harness with a per-request policy controller that decides when to reuse, when to speculate, and when to bypass the cache entirely.

---

## Key Results

### Main HF Evaluation (5 models, 10 datasets, 5 seeds)

| Engine | Mean Speedup | Waste | Hit Rate |
|--------|:-----------:|:-----:|:--------:|
| No cache | 1.000× | 0.000 | 0.000 |
| Reactive | 1.214× | 0.000 | 0.317 |
| Greedy | 1.221× | 0.000 | 0.320 |
| Strict reactive | 1.254× | 0.000 | 0.310 |
| Frequency spec. | 1.208× | 0.284 | 0.617 |
| ShadowKV | 1.287× | 0.264 | 0.606 |
| **ShadowKV++** | **1.365×** | **0.156** | **0.402** |

- Waste 41% below ShadowKV (0.156 vs 0.264)
- Significant on all 10 datasets ($p < 0.001$, nine datasets; $p < 10^{-4}$, one dataset)

### Runtime Evaluation (SGLang, LMCache, vLLM — Qwen2.5 1.5B–32B)

| Metric | Value |
|--------|-------|
| ShadowKV++ vs LMCache at 7B | +16.7% |
| ShadowKV++ at 32B over LMCache | +5.1% |
| vLLM APC+ShadowKV++ vs no-cache at 32B | +19.0% |
| GPU energy saving (vLLM at 32B) | ≈25% |

### Semantic Fidelity (KV cache reuse preserves output quality)

Evaluated across 5 architectures using DynamicCache.crop() at 75% shared prefix (float16).

| Model | KV Fidelity (ROUGE-L) | Verdict |
|-------|:---------------------:|:-------:|
| TinyLlama (1.1B) | **0.966** | ✅ Safe |
| Gemma 2B | **0.974** | ✅ Safe |
| Phi-3 Mini (3.8B) | **0.931** | ⚠️ Acceptable |
| GPT-2 (124M) | **0.876** | ⚠️ Acceptable |
| Qwen2.5 1.5B | **0.200** | ❌ Needs precision guard |

**Key finding**: KV cache reuse fidelity is **architecture-dependent** and **precision-dependent**. LLaMA-family and Gemma models achieve ROUGE-L > 0.96 in both float32 and float16. Qwen2.5 fails in float16 (ROUGE-L = 0.200) due to a precision–architecture interaction in its 28-layer attention stack. See [`docs/fidelity_deep_analysis.md`](docs/fidelity_deep_analysis.md).

### Controller Overhead

Plan() latency profiled across 1,000 synthetic requests (FakeBackend, CPU):

| Component | Mean Latency | Frequency |
|-----------|:-----------:|:---------:|
| Policy planning | 2.30 ms | 100% of requests |
| Semantic index query | 7.18 ms | 17% of requests |
| Amortised overhead | 1.2 ms | — |

Controller overhead is 0.5–2.3% of median GPU inference latency.

---

## Repository Layout

```
├── src/proactive_kv_cache/       Core engines, cache bank, controller, models
├── experiments/                  Benchmark scripts, evaluation, profiling
│   ├── run_fidelity_equiv.py     KV cache reuse fidelity pipeline
│   ├── eval_comprehensive.py     ROUGE-L and exact-match evaluator
│   ├── profile_plan.py           Controller Plan() latency profiler
│   ├── run_semantic_fidelity.py  Semantic fidelity measurement
│   └── ...
├── results/
│   ├── final_p100/               Canonical P100 result tree
│   ├── final_t4/                 Canonical T4 result tree
│   ├── fidelity/                 Fidelity experiment JSON results
│   │   ├── all_results.json      1,221 samples (5 models × 10 datasets)
│   │   ├── gpt2_results.json
│   │   ├── tinyllama_results.json
│   │   ├── qwen25_15b_results.json
│   │   ├── gemma2b_results.json
│   │   ├── phi3mini_results.json
│   │   ├── control/              Ratio=0.0 control (13 samples, 100% match)
│   │   └── qwen_ratios/          Qwen multi-ratio sweep (75/50/25%)
│   ├── RESULTS.md
│   ├── manifest.json
│   └── summary_by_engine.csv
├── docs/
│   ├── experimental_setup.md     Full methodology description
│   ├── results_table.md          Complete results with precision comparison
│   ├── fidelity_deep_analysis.md Root cause of Qwen's float16 failure
│   ├── semantic_fidelity.md      Comprehensive fidelity report
│   └── ...
├── runtime_experiments/          SGLang/LMCache/vLLM benchmark results
├── tests/                        Unit and regression tests
└── pyproject.toml
```

---

## Setup

Python 3.10+ recommended.

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
pip install pytest
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -e .
pip install pytest
```

---

## Validate

```bash
python -m pytest -q
```

Expected: `49 passed, 1 skipped`. The skipped test is an optional slow HF KV-correctness check.

---

## Fidelity Experiment (CPU)

```bash
python experiments/run_fidelity_equiv.py \
  --device cpu \
  --models tinyllama \
  --datasets samsum alpaca_eval banking77 \
  --n_samples 8 \
  --max_gen_tokens 64
```

### GPU (Colab, T4, ~1 hour for all 5 models × 10 datasets × 128 samples)

Upload [`experiments/fidelity_equiv_colab.ipynb`](experiments/fidelity_equiv_colab.ipynb) to Google Colab and run.

### Evaluate results

```bash
python experiments/eval_comprehensive.py
```

---

## Controller Overhead Profiling

```bash
python experiments/profile_plan.py
```

Runs 1,000 synthetic requests through the engine's Plan() and reports mean/P99 latency.

---

## Full HF Benchmark Reproduction

See [`docs/reproducing_results.md`](docs/reproducing_results.md) and
[`docs/runtime_experiments.md`](docs/runtime_experiments.md).

---

## Important Correctness Boundary

Exact-prefix KV reuse is semantics-preserving under causal attention. Approximate semantic KV reuse is not generally correctness-preserving. ShadowKV++ separates opportunity detection, utility scoring, execution admission, and backend correctness validation. See [`docs/semantic_fidelity.md`](docs/semantic_fidelity.md) for the full fidelity analysis.

---

## Citation

If you use this artifact, cite the associated ShadowKV++ manuscript.

---

## Acknowledgements

P100 experiments run using NVIDIA P100 GPU access provided by Prof. Sparsh Mittal and the Department of Electronics and Communication Engineering, IIT Roorkee.
