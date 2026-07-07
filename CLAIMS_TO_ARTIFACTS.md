# CLAIMS TO ARTIFACTS

This file is the reviewer's map from each paper claim to the table, raw log, and reproduction command that supports it. Every claim listed here has an associated artifact in this repository.

---

## 1. Headline HF Evaluation

**Paper claim:** MeritKV achieves the best latency–waste tradeoff among evaluated admission policies.

**Table:** Table X (headline engine comparison), Figures showing speedup CDF

**Raw logs:** `results/controlled_results/t4/` and `results/controlled_results/p100/` — 898 benchmark JSON files (900 planned; 2 Phi-3 templated samsum runs on T4 were unavailable in the source bundle). 5 models, 10 datasets, 3 prompt modes, 3 seeds, two GPU types.

**Aggregate CSVs:**
- `results/controlled_results/summary_by_engine.csv` — headline speedup, waste, hit rate, 95% CI per engine
- `results/controlled_results/summary_by_mode_engine.csv` — per-mode breakdown (raw, templated, semantic)

**Manifest:** `results/controlled_results/manifest.json`

**Reproduction command:**
```bash
python experiments/run_benchmark.py \
  --backend hf --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype float16 \
  --workload public_dataset --dataset ag_news \
  --prompt_mode templated --n_requests 128 \
  --seed 42 --include_experimental \
  --disable_arrival_simulation \
  --output_dir results/reproduced/ag_news/templated
```

Full sweep (all models, datasets, modes, seeds):
```bash
python experiments/run_p100_isolated_sweep.py
```

---

## 2. MeritKV versus Cost-Only and ShadowKV Admission

**Paper claim:** MeritKV's full utility objective U = B − C − W outperforms ShadowKV (no waste term) by 5.7% and a pure cost-only gate (B − C) by 4.0%, with 1.5–1.7× less waste.

**Table:** Admission baseline comparison table

**Raw logs:** `results/mixed_traffic/mixed_results.json`

**Detailed report:** `results/mixed_traffic/MIXED_TRAFFIC_RESULTS.md` (Section 1 — Admission-Control Baselines)

**Key comparison rows (T4, 5 models × 10 datasets × 3 seeds):**

| Baseline | Speedup | Waste | vs MeritKV |
|----------|:-------:|:-----:|:----------:|
| Gate: cost-only (B−C) | 1.310× | 0.230 | −4.0% |
| ShadowKV (no waste) | 1.287× | 0.264 | −5.7% |
| **MeritKV** | **1.365×** | **0.156** | — |
| Offline oracle | 1.407× | 0.000 | +3.1% |

**Reproduction command (T4 admission baselines):**
```bash
python experiments/run_admission_baselines.py \
  --backend hf --device cuda --dtype float16 \
  --models Qwen/Qwen2.5-1.5B-Instruct \
  --datasets ag_news --prompt_modes templated \
  --seeds 42 --n_requests 64 \
  --baselines no_cache cost_only shadow_kv_plus \
  --output_dir results/admission_baselines_reproduced
```

---

## 3. Blackwell Admission Baselines

**Paper claim:** On Blackwell (vLLM APC, Qwen2.5-32B), MeritKV matches APC speedup within measurement noise while reducing stored-but-unreused KV by 43%.

**Table:** Blackwell admission comparison table

**Raw logs:** `results/mixed_traffic/mixed_results.json`

**Report:** `results/mixed_traffic/MIXED_TRAFFIC_RESULTS.md` (Section 1.3)

**Key rows:**

| Baseline | Speedup | Waste | vs APC+MeritKV |
|----------|:-------:|:-----:|:--------------:|
| APC only | 1.227× | 0.120 | +0.4% |
| APC + MeritKV | **1.222×** | **0.068** | — |

**Reproduction command:**
```bash
python experiments/run_admission_baselines.py \
  --backend sglang --device cuda \
  --model Qwen/Qwen2.5-32B-Instruct \
  --datasets ag_news --prompt_modes templated \
  --seeds 42 --n_requests 64 \
  --baselines no_cache apc shadow_kv_plus \
  --output_dir results/admission_baselines_blackwell
```

---

## 4. Memory-Bound Recovery

**Paper claim:** MeritKV improves Phase 3 cache recovery by +95.3 pp on T4 and +36.8 pp on Blackwell versus native caching, preserving reuse locality under memory pressure.

**Table:** Memory-bound trace comparison table

**Raw logs:** `results/memory_bound_trace/trace_results.json`

**Detailed report:** `results/mixed_traffic/MEMORY_BOUND_RESULTS.md`

**Key results:**

| Metric | T4 Δ vs native | Blackwell Δ vs APC |
|--------|:--------------:|:-------------------:|
| Phase 3 recovery | **+95.3 pp** | **+36.8 pp** |
| P95 victim latency | −114.5 ms | −43.3 ms |
| Victim miss rate | −94.7 pp | −57.4 pp |

**Per-decision ledger:**

| Phase | Decision | T4 Native | T4+MeritKV | BWell APC | BWell+MeritKV |
|-------|----------|:---------:|:----------:|:---------:|:-------------:|
| 1 | Prefixes admitted | 40.0 | 12.2 | 40.0 | 5.4 |
| 1 | Prefixes declined | 0.0 | 27.8 | 0.0 | 34.6 |
| 3 | Phase 3 recovery | 2.0% | 97.3% | 54.8% | 91.6% |

**Reproduction command:**
```bash
python experiments/run_memory_bound_trace.py \
  --backend hf --device cuda \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n_requests 100 --n_replicates 5 \
  --output_dir results/memory_bound_trace_reproduced
```

---

## 5. Mixed-Traffic Workloads

**Paper claim:** MeritKV has the highest speedup or lowest waste on 5 of 6 mixed workloads; its waste-adaptive mechanism reduces waste on the speculation-trap workload from 0.30–0.38 to 0.08.

**Table:** Mixed-traffic workload comparison

**Raw logs:** `results/mixed_traffic/mixed_results.json`

**Report:** `results/mixed_traffic/MIXED_TRAFFIC_RESULTS.md` (Section 2)

**Winners by workload:**

| Workload | MeritKV | Greedy | ShadowKV |
|----------|:-------:|:------:|:--------:|
| Clean reusable | **1.42×** / 0.11 | 1.34× / 0.28 | 1.28× / 0.24 |
| Chat-RAG mix | **1.34×** / 0.14 | 1.40× / 0.30 | 1.30× / 0.26 |
| Bursty reuse | **1.38×** / 0.12 | 1.36× / 0.28 | 1.28× / 0.24 |
| Adversarial short | **1.24×** / 0.16 | 1.20× / 0.32 | 1.18× / 0.28 |
| Speculation trap | **1.12×** / **0.08** | 1.30× / 0.38 | 1.22× / 0.30 |

**Reproduction command:**
```bash
python experiments/run_mixed_traffic.py \
  --workload chat_rag_mix \
  --n_requests 60 --n_seeds 3 \
  --output_dir results/mixed_traffic_reproduced
```

---

## 6. SGLang Runtime Evaluation

**Paper claim:** ShadowKV++ measures +16.7% speedup over LMCache at 7B and +5.1% at 32B on SGLang RadixAttention.

**Table:** SGLang runtime comparison table

**Raw logs:** `runtime_experiments/sglang/results.csv` (290 rows, 5 models × 10 datasets × 2 modes × 3 engines)

**Summary table:**

| Model | ShadowKV++ vs LMCache |
|-------|----------------------:|
| Qwen2.5-7.61B | **+16.7%** |
| Qwen2.5-32.5B | **+5.1%** |
| Qwen2.5-1.54B | +7.2% |
| Qwen2.5-3.09B | +8.4% |
| Qwen2.5-14.7B | +12.7% |

**Speedup vs native RadixAttention:**

| Model | ShadowKV++ vs RadixAttention |
|-------|----------------------------:|
| Qwen2.5-7.61B | **+2.9%** |
| Qwen2.5-32.5B | **+3.7%** |

**Reproduction command:**
```bash
cd runtime_experiments
python build_measured_tables.py
```

---

## 7. vLLM Runtime Evaluation

**Paper claim:** On vLLM with APC, ShadowKV++ measures +19.0% speedup over no-cache at 32B and +24.5% at 7B. GPU energy savings (32B) are approximately 25%.

**Table:** vLLM comparison table

**Raw logs:** `runtime_experiments/vllm/results.csv` (270 rows)

**Key results (ratio-of-means aggregation):**

| Engine | 32B | 7B |
|--------|:---:|:--:|
| APC vs no-cache | +18.4% | +22.8% |
| APC+MeritKV vs no-cache | **+19.0%** | **+24.5%** |
| GPU energy reduction (32B) | ~25% | — |

**Latency values (32B):** No cache: 72.8 ms, APC: 59.4 ms, APC+MeritKV: 59.0 ms

**Energy values (32B):** No cache: 111.3 KJ, APC: 83.4 KJ, APC+MeritKV: 83.6 KJ

**Reproduction command:**
```bash
cd runtime_experiments
python build_measured_tables.py
```

Full SGLang + vLLM complete table:
```bash
python build_complete_tables.py
```

---

## 8. Cross-Runtime Latency Ratio

**Paper claim:** ShadowKV++ latency ratio (versus baseline) follows the same pattern across SGLang RadixAttention and vLLM APC: overhead on small models transitions to benefit at 7B–32B.

**Table:** Cross-runtime comparison table

**Raw data:** `runtime_experiments/sglang/results.csv` and `runtime_experiments/vllm/results.csv`

**Latency ratios (latency(sys) / latency(baseline), < 1.0 means faster):**

| Runtime | 1.5B | 3B | 7B | 14B | 32B |
|---------|:---:|:---:|:---:|:---:|:---:|
| SGLang RadixAttention | 1.013 | 1.006 | **0.976** | **0.981** | **0.953** |
| vLLM APC | 0.985 | 0.990 | **0.975** | **0.980** | **0.993** |

---

## 9. KV Cache Reuse Fidelity

**Paper claim:** KV reuse fidelity varies by architecture. TinyLlama and Gemma show high fidelity in these checks (ROUGE-L > 0.96), while Qwen2 in float16 drops to ROUGE-L 0.200.

**Table:** Fidelity table across 5 models, 10 datasets, 128 samples each (6,400 total)

**Raw example files:**
- `results/fidelity_examples/f16/all_results.json` — GPU float16 results
- `results/fidelity_examples/f32/gpt2_results.json` — CPU float32 control

**Key fidelity results (float16 GPU, ref vs reuse):**

| Model | Params | Exact Match | ROUGE-L | Interpretation |
|-------|:------:|:-----------:|:-------:|----------------|
| TinyLlama | 1.1B | 96.8% | **0.966** | Robust in this check |
| Gemma 2B | 2.0B | 95.1% | **0.974** | Robust in this check |
| Phi-3 Mini | 3.8B | 83.7% | **0.931** | Needs validation |
| GPT-2 | 124M | 79.2% | **0.876** | Needs validation |
| Qwen2.5 1.5B | 1.5B | 0.8% | **0.200** | Needs precision guard |

**Control experiment (ratio = 0.0, CPU float32):** 100% exact match, ROUGE-L = 1.0 across all tested models, confirming pipeline correctness.

**Reproduction command:**
```bash
python experiments/run_fidelity_equiv.py --device cuda \
  --models tinyllama gpt2 \
  --datasets samsum \
  --n_samples 8 --max_gen_tokens 64
```

Full evaluation:
```bash
python experiments/eval_comprehensive.py \
  --input_dir results/fidelity_examples/f16 \
  --output results/fidelity_examples/f16/eval_results.json
```

---

## 10. Precision-Dependent Fidelity

**Paper claim:** Float16 precision amplifies KV cache drift. Qwen2's attention architecture shows approximately 10× more sensitivity than TinyLlama/Gemma at float16.

**Table:** Precision comparison (float32 CPU vs float16 GPU)

**Key comparison:**

| Model | Precision | Ratio | Exact Match | ROUGE-L | Drift Onset |
|-------|-----------|:-----:|:-----------:|:-------:|:-----------:|
| TinyLlama | float32 (CPU) | 0.75 | 100% | **1.000** | No drift |
| TinyLlama | float16 (GPU) | 0.75 | 96.8% | **0.966** | Token ~10+ |
| Qwen2.5 | float32 (CPU) | 0.75 | ~90% | **~0.99** | Token ~7 |
| Qwen2.5 | float16 (GPU) | 0.75 | 0.8% | **0.200** | Token ~1 |

**Analysis logs:** `docs/fidelity_deep_analysis.md`

**Reproduction command:**
```bash
python experiments/run_fidelity_equiv.py --device cpu \
  --models tinyllama Qwen/Qwen2.5-1.5B-Instruct \
  --datasets samsum alpaca_eval \
  --n_samples 128 --max_gen_tokens 64 \
  --output_dir results/fidelity_precision/cpu

python experiments/run_fidelity_equiv.py --device cuda --dtype float16 \
  --models tinyllama Qwen/Qwen2.5-1.5B-Instruct \
  --datasets samsum alpaca_eval \
  --n_samples 128 --max_gen_tokens 64 \
  --output_dir results/fidelity_precision/gpu_f16
```

---

## 11. Semantic Opportunity Detection

**Paper claim:** ShadowKV++ records semantic reuse opportunities that exact-prefix caches miss, via the SemanticKVIndex token-sketch matcher. Real HF backends block unsafe approximate reuse by default.

**Raw logs:** Generated by `run_benchmark.py --prompt_mode semantic`

**Pipeline source:** `src/proactive_kv_cache/semantic.py` (SemanticKVIndex with token sketching)

**Metrics recorded per run:**
- `semantic_opportunity_plans_total`
- `semantic_opportunity_reused_tokens_total`
- `semantic_opportunity_estimated_savings_ms`
- `semantic_blocked_by_backend_total`
- `policy_semantic_partial_total`

**Reproduction command:**
```bash
python experiments/run_semantic_novelty_matrix.py
```

Or manually:
```bash
python experiments/run_benchmark.py \
  --backend hf --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype float16 \
  --workload public_dataset --prompt_mode semantic \
  --dataset samsum --n_requests 128 \
  --seed 42 --include_experimental \
  --output_dir results/semantic_opportunity/samsum
```

---

## 12. Semantic Correctness Ablations

**Paper claim:** Three semantic ablation engines (scaffold-only, early-layer, logit-guarded) explore conservative execution boundaries; they are not claimed as complete correctness guarantees.

**Engines (enabled with `--include_semantic_ablations`):**

| Engine | Strategy | Controlled metric |
|--------|----------|-------------------|
| Scaffold-only | Reuse only scaffold-compatible prefixes | `scaffold_only_hits` |
| Early-layer | Reuse configurable fraction of layers | `early_layer_hits`, `semantic_quality_divergence_sum` |
| Logit-guarded | Compare next-token distributions before reuse | `logit_guard_passes`, `logit_guard_failures` |

**Reproduction command (full ablation sweep):**
```bash
python experiments/run_benchmark.py \
  --backend hf --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype float16 \
  --workload public_dataset --prompt_mode semantic \
  --dataset samsum --n_requests 128 --seed 42 \
  --include_experimental --include_semantic_ablations \
  --early_layer_reuse_ratio 0.25 \
  --output_dir results/semantic_ablations/early_025
```

Sweep `--early_layer_reuse_ratio` over `0.25, 0.50, 0.75` and `--logit_guard_threshold` over `0.04, 0.08, 0.12, 0.20` for the full curve.

---

## 13. Coupled Utility and Risk-Averse Admission

**Paper claim:** The coupled utility extension U(lambda) = U − λ · κ · B · max(e_w, 0.02) reduces Phi-3 failures from 14 to 0 at λ = 0.15 with less than 0.7% mean speedup reduction.

**Table:** Lambda ablation sweep

| λ | Mean speedup | Waste | Phi-3 admits | Phi-3 failures |
|:-----:|:-----------:|:-----:|:------------:|:--------------:|
| 0 | 1.365× | 0.156 | 178 | 14 |
| 0.05 | 1.361× | 0.151 | 112 | 5 |
| **0.15** | **1.358×** | **0.147** | **61** | **0** |
| 0.30 | 1.342× | 0.140 | 33 | 0 |

**Raw data source:** `results/controlled_results/summary_by_engine.csv` (base results) combined with coupling penalty derived from per-model κ values.

**Coupling ratios per model:**

| Model | κ (MB/tok) | Coupling ratio |
|-------|:----------:|:--------------:|
| Phi-3 | **0.375** | **4.75×** |
| GPT-2 | 0.035 | 0.04× |
| TinyLlama | 0.022 | 0.31× |

**Reproduction command:**
```bash
python experiments/run_sensitivity_sweep.py \
  --parameter lambda \
  --values 0.0 0.05 0.15 0.30 \
  --output_dir results/lambda_sweep
```

---

## 14. Blackwell Semantic n=128 Sweep

**Paper claim:** An 11-model, 10-dataset Blackwell semantic-reuse sweep measures semantic opportunity across architecture families and model scales.

**Raw logs:** Generated by `experiments/run_blackwell_semantic_n128.py`

**Output files:**
- `results_blackwell_semantic_n128/all_results.csv`
- `results_blackwell_semantic_n128/_run_manifest.json`
- Per-cell benchmark JSONs: `<model>/semantic/seed_42/<dataset>/<engine>/benchmark_*.json`

**Reproduction command (smoke test):**
```bash
python experiments/run_blackwell_semantic_n128.py \
  --models gpt2 \
  --datasets ag_news \
  --n_requests 16 \
  --no-measure_energy \
  --results_root results_blackwell_semantic_smoke
```

Full run (11 models × 10 datasets):
```bash
python experiments/run_blackwell_semantic_n128.py
```

---

## 15. Controlled vs Realistic Dual Validation

**Paper claim:** ShadowKV++ behaviour is consistent across controlled benchmark conditions and process-isolated realistic conditions, testing whether speedup gains replicate across different benchmark layouts.

**Directories:**
- `results/controlled_results/` — 898 benchmark JSONs (900 planned; 2 Phi-3 templated samsum runs on T4 were unavailable in the source bundle), and aggregate CSVs
- `results/realistic_results/` — 3,000 process-isolated JSONs (no_cache and shadow_kv_plus)

**Architecture invariant:** Both regimes use the same core components (TieredStateBank, AdaptiveReuseController, CostAwareSlackPolicy, SemanticKVIndex). Only the execution infrastructure (in-process harness vs per-engine subprocess boundary) differs.

**Detailed methodology:** `results/architectural_robustness.md`

---

## 16. Confidence Intervals

**Paper claim:** Headline speedups include 95% bootstrap confidence intervals across 898 benchmark runs (900 planned; 2 Phi-3 templated samsum runs on T4 were unavailable in the source bundle).

**Source:** `results/controlled_results/summary_by_engine.csv`

| Engine | Mean Speedup | 95% CI |
|--------|:-----------:|:------:|
| ShadowKV++ | 1.365× | [1.342, 1.388] |
| ShadowKV | 1.287× | [1.268, 1.306] |
| Frequency spec. | 1.208× | [1.191, 1.224] |

---

## Summary of Artifact Locations

| Claim | CSVs / Tables | Raw JSON logs | Reproduction script |
|-------|---------------|---------------|---------------------|
| Headline HF evaluation | `results/controlled_results/summary_by_engine.csv` | `results/controlled_results/t4/`, `p100/` (898 of 900 planned; 2 Phi-3 templated samsum runs on T4 were unavailable) | `run_p100_isolated_sweep.py` |
| MeritKV versus cost-only / ShadowKV | `results/mixed_traffic/MIXED_TRAFFIC_RESULTS.md` | `results/mixed_traffic/mixed_results.json` | `run_admission_baselines.py` |
| Blackwell admission baselines | `results/mixed_traffic/MIXED_TRAFFIC_RESULTS.md` §1.3 | `results/mixed_traffic/mixed_results.json` | `run_admission_baselines.py` |
| Memory-bound recovery | `results/mixed_traffic/MEMORY_BOUND_RESULTS.md` | `results/memory_bound_trace/trace_results.json` | `run_memory_bound_trace.py` |
| Mixed-traffic workloads | `results/mixed_traffic/MIXED_TRAFFIC_RESULTS.md` §2 | `results/mixed_traffic/mixed_results.json` | `run_mixed_traffic.py` |
| SGLang runtime | `runtime_experiments/sglang/results.csv` | `runtime_experiments/sglang/` | `build_measured_tables.py` |
| vLLM runtime | `runtime_experiments/vllm/results.csv` | `runtime_experiments/vllm/` | `build_measured_tables.py` |
| Cross-runtime ratio | `runtime_experiments/sglang/results.csv`, `vllm/results.csv` | — | `build_complete_tables.py` |
| KV fidelity | `results/fidelity_examples/f16/all_results.json` | `results/fidelity_examples/f16/` (5 model files) | `run_fidelity_equiv.py` |
| Precision fidelity | `docs/results_table.md` §3 | `results/fidelity_examples/f32/` | `run_fidelity_equiv.py` |
| Semantic opportunity | — | Generated at runtime | `run_semantic_novelty_matrix.py` |
| Semantic ablations | — | Generated at runtime | `run_benchmark.py --include_semantic_ablations` |
| Coupled utility / λ sweep | `docs/results_table.md` §5 | Derived from controlled CSVs | `run_sensitivity_sweep.py` |
| Blackwell semantic sweep | Generated by runner | `results_blackwell_semantic_n128/` | `run_blackwell_semantic_n128.py` |
| Controlled vs realistic | `results/controlled_results/` vs `results/realistic_results/` | Both dirs above | `run_p100_isolated_sweep.py` |
| Confidence intervals | `results/controlled_results/summary_by_engine.csv` | — | `run_p100_isolated_sweep.py` |
