# Architectural Robustness: Controlled and Realistic Validation

ShadowKV++ is evaluated under two complementary result regimes. The purpose is not to present one best-case number, but to show how the same per-request utility controller behaves under controlled benchmark conditions and under cleaner deployment-style process boundaries.

| Regime | Files | Purpose |
|--------|-------|---------|
| Controlled | `results/controlled_results/` | Main academic comparison across engines, models, datasets, prompt modes, and seeds |
| Realistic | `results/realistic_results/` | Process-isolated no-cache versus ShadowKV++ traces for deployment sanity checks |

## Architecture Invariants

The controller logic is the same across both regimes:

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

Core components:

| Component | Responsibility |
|-----------|----------------|
| TieredStateBank | Prefix-keyed radix trie with reuse statistics and memory accounting |
| AdaptiveReuseController | Per-request benefit, cost, and waste scoring |
| CostAwareSlackPolicy | Speculation priority by expected utility gain |
| SemanticKVIndex | Lightweight semantic opportunity detection |
| Raw-mode gate | Conservative admission behavior for unstructured prompts |
| Telemetry | Per-request policy and cache accounting |

## Controlled Results

The controlled bundle is the primary source for aggregate paper-style results:

```text
results/controlled_results/
  summary_by_engine.csv
  summary_by_mode_engine.csv
  manifest.json
  t4/**/benchmark_*.json
  p100/**/benchmark_*.json
```

These runs compare multiple in-repository engines on the same benchmark harness. They are the right source for headline speedup, p95, hit-rate, waste, and confidence-interval summaries.

## Realistic Results

The realistic bundle contains process-isolated traces for the most important deployment comparison:

```text
results/realistic_results/
  no_cache/**/benchmark_*.json
  shadow_kv_plus/**/benchmark_*.json
```

Use these files when checking whether the policy still behaves sensibly when each engine starts from a cleaner process boundary. They are not a replacement for the controlled aggregate CSVs; they are a sanity check that the measured gains are not only artifacts of one in-process benchmark layout.

## Why Both Regimes Matter

Controlled results answer the research question: does per-request utility admission beat simpler cache strategies when everything is measured in one benchmark harness?

Realistic results answer the deployment question: does the same policy avoid obvious negative-utility behavior when measured with cleaner process boundaries and a direct no-cache comparison?

Together, they support a more honest claim: ShadowKV++ improves reuse decisions by admitting only requests with positive expected utility, while exposing cases where bypass is better than reuse.

## Interpretation Rules

- Speedup should be read together with waste and hit rate.
- A high hit rate is not automatically good if the matched prefixes are too short or the controller pays too much overhead.
- Raw-mode gains should be interpreted as bypass and overhead-avoidance gains unless exact reuse is directly shown.
- Semantic matches are opportunities unless the backend and guard path actually execute correctness-bounded reuse.
- Approximate semantic KV substitution should not be described as generally safe without model- and precision-specific evidence.

## Current Result Layout

```text
results/
  controlled_results/     # Controlled T4/P100 benchmark outputs and CSV summaries
  realistic_results/      # Process-isolated no_cache and shadow_kv_plus outputs
  fidelity_examples/      # Per-sample KV reuse fidelity examples
  sweep_timing/           # Small timing/smoke outputs
  RESULTS.md              # Public result-bundle guide
```
