# Architectural Robustness: Dual-Validation Methodology

## Overview

MeritKV is evaluated under two complementary regimes. The point is not to present a single best-case number, but to show how the same cache-reuse policy behaves under both controlled research conditions and more realistic execution boundaries.

| Regime | Conditions | Purpose |
|---|---|---|
| Controlled | Dedicated benchmark runs with the full multi-engine harness and aggregated T4/P100 outputs | Reproducible academic baseline and ceiling-style comparison |
| Realistic | Process-isolated result files for deployment-style no-cache versus MeritKV checks | Practical sanity check under cleaner per-engine process boundaries |

The core architecture is the same in both regimes. What changes is the execution infrastructure around the engines.

## 1. Architecture Invariants

The same policy components are used across the validation regimes.

### Engine Hierarchy

```text
BaseEngine
  +-- NoCacheEngine
  +-- NativePrefixCachingEngine
  +-- ReactivePrefixCacheEngine
  |   +-- StrictReactivePrefixCacheEngine
  |   +-- GreedyPrefixCacheEngine
  |   +-- FrequencySpeculativeEngine
  |   +-- ShadowKVEngine
  |       +-- ShadowKVPlusEngine
```

### Core Components

| Component | Responsibility | Stable Across Regimes |
|---|---|---|
| TieredStateBank | Prefix-keyed bank with EWMA frequency, hit tracking, and support metrics | Yes |
| AdaptiveReuseController | Waste-aware utility planner using benefit, cost, and waste | Yes |
| CostAwareSlackPolicy | Speculation priority by expected utility gain | Yes |
| SemanticKVIndex | Lightweight token-sketch similarity for paraphrase opportunity | Yes |
| Raw-mode gate | Conservative admission behavior for unstructured prompts | Yes |
| Logit-guard reuse | Output-distribution check before guarded semantic reuse variants | Yes |

The architectural components do not change between result folders. The difference is how measurements are run and aggregated.

## 2. Controlled Versus Realistic Results

### Controlled Results

The controlled result folder is:

```text
results/controlled_results/
```

It contains T4 and P100 benchmark JSONs plus aggregate CSVs:

```text
summary_by_engine.csv
summary_by_mode_engine.csv
```

These files are the primary source for paper-style aggregate speedup, confidence interval, p95, hit-rate, and waste numbers.

### Realistic Results

The realistic result folder is:

```text
results/realistic_results/
```

It contains process-isolated JSON outputs organized by engine:

```text
realistic_results/no_cache/...
realistic_results/shadow_kv_plus/...
```

These files are useful for checking behavior when each engine run starts from a cleaner process boundary. They should not be mixed into the controlled aggregate CSVs unless a separate aggregation script explicitly does so.

## 3. Validation Principles

### Honest Baselines

Every performance claim should compare a reuse engine against the matching no-cache condition for the same model, dataset, prompt mode, seed, hardware, backend, and precision.

```text
speedup = latency(no_cache) / latency(strategy)
```

### Cost Transparency

Cache operations are treated as real work:

- Reactive storage includes prefill and KV materialization cost.
- Speculation includes idle-time precompute cost.
- Waste is measured when speculative work is not reused.
- Hit-rate is not enough; latency and waste determine whether reuse helped.

### Failure Mode Capture

The framework is designed to expose cases where caching is net-negative:

- Reuse can be bypassed when expected utility is poor.
- Waste is reported directly.
- Raw prompts use conservative admission behavior.
- Semantic reuse is treated as an opportunity that needs safety boundaries, not as an automatic correctness-preserving optimization.

## 4. Why This Matters

For reproducibility, the controlled results provide a stable aggregate benchmark. For deployment reasoning, the realistic results show whether the idea survives cleaner per-engine process boundaries. Keeping both regimes visible makes the paper more honest: the reader can separate best-case controlled behavior from deployment-style sanity checks.

## 5. Key Takeaways

1. Exact-prefix reuse is the most stable case.
2. Utility-aware admission matters because hit-rate alone can be misleading.
3. Speculation helps only when future support is strong enough to pay for precompute.
4. Raw prompts need conservative gating because reuse structure is weaker.
5. Semantic reuse must be reported with correctness boundaries and should not be treated as blind prefix caching.

## 6. Results Structure

```text
results/
  controlled_results/     # T4/P100 controlled aggregate benchmark outputs
  realistic_results/      # Process-isolated no_cache and shadow_kv_plus outputs
  fidelity_examples/      # Per-sample fidelity examples
  sweep_timing/           # Small timing smoke outputs
```

Raw benchmark JSON files generally follow this naming pattern:

```text
benchmark_hf_<model>_<workload>_<dataset_or_mode>_<device>.json
```

Use `results/RESULTS.md` for the current bundle overview.
