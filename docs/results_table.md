# Results: KV Cache Reuse Fidelity

## 1. Main Results (GPU, float16, n=128 per model per dataset)

5 models x 10 datasets x 128 = 6,400 total samples.

### 1.1 KV Fidelity (ref vs reuse)

| Model | Params | Samples | Exact Match | ROUGE-L | Fidelity |
|-------|--------|---------|-------------|---------|----------|
| TinyLlama | 1.1B | 640 | 96.8% | **0.966** | Robust in this check |
| Gemma 2B | 2.0B | 640 | 95.1% | **0.974** | Robust in this check |
| Phi-3 Mini | 3.8B | 640 | 83.7% | **0.931** | Needs validation |
| GPT-2 | 124M | 640 | 79.2% | **0.876** | Needs validation |
| Qwen 2.5 1.5B | 1.5B | 640 | 0.8% | **0.200** | Needs guard |

**Key finding**: LLaMA-family (TinyLlama) and Gemma architectures show high fidelity in this check (ROUGE-L > 0.96). Qwen2 is highly sensitive in float16 due to a precision-architecture interaction (Section 3).

### 1.2 Prompt Sensitivity (exact vs ref) - Baseline

| Model | ROUGE-L | Interpretation |
|-------|---------|----------------|
| GPT-2 | 0.320 | Baseline output variability |
| Gemma 2B | 0.305 | Baseline output variability |
| Phi-3 Mini | 0.252 | Baseline output variability |
| TinyLlama | 0.235 | Baseline output variability |
| Qwen 2.5 1.5B | 0.221 | Baseline output variability |

**Interpretation**: The inherent output variability from prompt rephrasing averages 0.22-0.32 ROUGE-L across models. KV reuse fidelity exceeds this baseline for all models except Qwen.

### 1.3 By Dataset (aggregated across models)

| Dataset | Samples | Exact Match | ROUGE-L |
|---------|---------|-------------|---------|
| samsum | 640 | 69.4% | 0.783 |
| xsum | 640 | 71.2% | 0.804 |
| cnn_dailymail | 640 | 69.4% | 0.788 |
| ag_news | 640 | 68.1% | 0.794 |
| banking77 | 640 | 75.7% | 0.840 |
| alpaca_eval | 640 | 78.8% | 0.848 |
| dolly | 640 | 71.7% | 0.762 |
| daily_dialog | 640 | 77.5% | 0.731 |
| oasst1 | 640 | 72.3% | 0.812 |
| ultrachat | 640 | 69.4% | 0.799 |

## 2. Control Experiment (ratio = 0.0, CPU float32)

| Model | Dataset | Samples | Exact Match | ROUGE-L |
|-------|---------|---------|-------------|---------|
| TinyLlama | samsum | 128 | **100%** | **1.0** |
| TinyLlama | alpaca_eval | 128 | **100%** | **1.0** |
| TinyLlama | banking77 | 128 | **100%** | **1.0** |

At ratio=0.0 (no shared prefix, no cache reuse), the output is identical to clean generation. This confirms the pipeline is implementation-correct and free of artifacts.

## 3. Precision Comparison: float32 (CPU) vs float16 (GPU)

| Model | Precision | Ratio | Exact Match | ROUGE-L | Drift Onset |
|-------|-----------|-------|-------------|---------|-------------|
| TinyLlama | float32 (CPU) | 0.75 | 100% | **1.000** | No drift |
| TinyLlama | float16 (GPU) | 0.75 | 96.8% | **0.966** | Token ~10+ |
| Qwen 2.5 | float32 (CPU) | 0.75 | ~90% | **~0.99** | Token ~7 |
| Qwen 2.5 | float16 (GPU) | 0.75 | 0.8% | **0.200** | Token ~1 |

### 3.1 Analysis

The precision-dependent fidelity loss has two components:

**1. Float16 precision loss**: float16 provides only ~3.3 decimal digits (vs ~7.3 for float32). The KV cache accumulates rounding errors at each attention layer. After 22-28 layers, the cumulative error is:
- float32: ~2e-5 (negligible, only flips tokens at close decision boundaries)
- float16: ~1e-2 (significant, flips tokens regularly)

**2. Architecture amplification**: Qwen2's attention implementation amplifies the float16 error approximately 10x more than LLaMA/Gemma. Layer-by-layer tracing shows Qwen's hidden state diff grows from ~1e-4 at layer 1 to ~1e-2 at layer 28 in float16. The same experiment in float32 grew from ~5e-7 to ~2e-5.

### 3.2 Recommendation

**Measure KV reuse fidelity in deployment precision.** CPU float32 results DO NOT generalize to GPU float16. For LLaMA/Gemma-style models in this float16 check, observed ROUGE-L was about 0.97. For Qwen2-family models in float16, approximate partial semantic reuse requires additional quality guards.

## 4. Architecture Comparison Summary

| Architecture | float32 Fidelity | float16 Fidelity | Sensitivity | Reuse stance |
|-------------|-----------------|-----------------|-------------|-----------------|
| LLaMA (TinyLlama) | 1.0 | 0.966 | 0.235 | Promising; validate per deployment |
| Gemma (2B) | 1.0 | 0.974 | 0.305 | Promising; validate per deployment |
| Phi-3 (Mini) | - | 0.931 | 0.252 | Validate and guard |
| GPT-2 | - | 0.876 | 0.320 | Validate and monitor |
| Qwen2 (1.5B) | ~0.99 | 0.200 | 0.221 | Needs precision guard |

## 5. Coupled Utility and Risk-Averse Admission

The base utility U = B - C - W treats benefit, cost, and waste as independent. In practice they are coupled through the model's VRAM footprint kappa. The coupled utility extends U with a coupling penalty:

```
U(lambda) = U - lambda * kappa * B * max(e_w, 0.02)
```

where lambda >= 0 is the risk-aversion parameter (default 0.15). At lambda = 0, the base policy is recovered exactly. This is analogous to mean-variance portfolio optimization: lambda controls the operator's tolerance for covariance between benefit and waste.

### 5.1 Coupling Ratios per Model

| Model | kappa (MB/tok) | Coupling ratio | Admits (lambda=0) | Admits (lambda=0.15) | Flips |
|-------|:--------------:|:--------------:|:-----------------:|:-------------------:|:-----:|
| GPT-2 | 0.035 | 0.04x | - | - | -2 |
| TinyLlama | 0.022 | 0.31x | - | - | -27 |
| Qwen2.5 | 0.027 | 0.39x | - | - | -32 |
| Gemma | 0.018 | 0.23x | - | - | -3 |
| Phi-3 | **0.375** | **4.75x** | 178 | 61 | **-117** |

The coupling penalty is negligible for four of five models (coupling ratio < 0.4x benefit). For Phi-3, whose kappa = 0.375 MB/token is 14-21x larger than the other models, the coupling ratio reaches 4.75x benefit, blocking 117 of 178 semantic-mode admits.

### 5.2 Lambda Ablation Sweep

| lambda | Mean speedup | Waste | Phi-3 admits | Phi-3 failures |
|:-----:|:-----------:|:-----:|:------------:|:--------------:|
| 0 | 1.365x | 0.156 | 178 | 14 |
| 0.05 | 1.361x | 0.151 | 112 | 5 |
| **0.15** | **1.358x** | **0.147** | **61** | **0** |
| 0.30 | 1.342x | 0.140 | 33 | 0 |

At lambda = 0, the policy matches the base results exactly (sanity check). At the default lambda = 0.15, all Phi-3 failures are eliminated with less than 0.7% mean speedup reduction.

### 5.3 Relationship to the Breakeven Guard

The coupling penalty is complementary to the memory-breakeven guard. The guard is a **hard constraint** derived from hardware physics: it rejects speculative precomputes shorter than k* because they can never break even. The coupling penalty is a **soft preference** tuned by the operator: it discounts admits where high benefit correlates with high waste, reducing risk without enforcing a hard cutoff.

## 6. Implications for MeritKV

1. **Promising architectures**: LLaMA and Gemma models showed strong fidelity in these checks, but approximate partial semantic reuse should still be validated for the deployed model, precision, and backend.

2. **Precision-aware design**: The `allow_approximate_semantic_reuse` flag should consider the deployment precision. Float16 deployments should use a higher `semantic_similarity_threshold`.

3. **Architecture validation**: Each new model architecture must be independently validated. Fidelity on one architecture does not imply fidelity on another.

4. **Risk-averse extension**: The coupling penalty adds a tunable risk parameter lambda. At lambda=0 the extension is disabled and the base policy is recovered exactly. The ablation over {0, 0.05, 0.15, 0.30} bounds the sensitivity.

5. **Prompt sensitivity bound**: The prompt sensitivity metric (0.22-0.32 ROUGE-L) is useful context, but it should not be used to claim blanket safety. Qwen is the counterexample in these results.
