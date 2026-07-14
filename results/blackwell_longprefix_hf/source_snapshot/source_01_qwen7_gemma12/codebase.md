# Codebase Map

Created: 2026-06-27T03:05:52.408Z
Last updated: 2026-06-27

This file is maintained by the stratum manager and worker agents. It is the shared, up-to-date reference for understanding this project without repeatedly re-researching the same codebase.

## Purpose

**ShadowKV v10 (proactive-kv-cache v0.3.0)** — A research prototype for adaptive prefix/KV cache reuse in LLM (Large Language Model) serving. The project explores strategies for proactively caching and reusing key-value (KV) prefixes across inference requests to reduce latency and computational cost. It features multiple caching engines, policy-driven speculation, semantic similarity matching, tiered hierarchical storage (GPU/CPU/Disk), online utility estimation, and a full test/benchmark suite.

## Architecture

The system is organized as a Python package (`proactive_kv_cache`) under `src/`, with the following layers:

### Core Cache Layer (`cache.py`)
- **`TieredStateBank`**: The central hierarchical KV cache with GPU, CPU, and Disk tiers. Uses a trie for prefix matching, an exact-match hash map for O(1) lookups, and an LRU recent-match cache. Tracks per-prefix frequency (EMA-decayed), observation count, continuation tokens, and branching factor. Supports speculation tracking (wasted vs. useful precomputes).
- **`CacheEntry`**: Dataclass holding prefix tokens, KV cache blob, frequency, tier, timing metadata, and speculation state.

### Engine Layer (`engines.py`)
Multiple engine implementations (28K+ lines) providing different caching strategies:
- **`NoCacheEngine`** — Baseline without caching.
- **`NativePrefixCachingEngine`** — Baseline using native prefix caching (placeholder-based).
- **`RuntimeNativeCacheEngine`** — Named native-runtime cache baseline (for vLLM APC, SGLang RadixAttention, LMCache).
- **`AdmissionControlledRuntimeCacheEngine`** — Native runtime baseline gated by ShadowKV++ admission policy.
- **`ReactivePrefixCacheEngine`** — Basic reactive caching (store observed prefixes, reuse on match).
- **`StrictReactivePrefixCacheEngine`** — Stricter variant with higher minimum prefixes and coverage ratios.
- **`GreedyPrefixCacheEngine`** — Minimally restrictive caching.
- **`FrequencySpeculativeEngine`** — Background thread speculatively prefills frequent prefixes during idle time.
- **`ShadowKVEngine`** — Idle-time precompute with `CostAwareSlackPolicy`, tier promotion, adaptive speculation controller.
- **`ShadowKVPlusEngine`** — Full policy-driven engine with semantic retrieval (via `TokenSketcher`/`SemanticKVIndex`), utility-based admission (`AdaptiveReuseController`), fine-grained KV reuse, sandbox safety checks, raw conservative gate, scaffold bypass, and semantic ablations (scaffold_only, early_layer, logit_guard).
- **`ShadowKVPlusLiteEngine`** — Low-overhead exact-prefix serving baseline (Phase 1), no background speculation, no semantic retrieval, utility-based admission.

### Policy Layer
- **`base_policy.py`** — `ReusePlan` (strategy, depth, confidence, utility), `PolicyController` ABC.
- **`policy.py`** — `SpeculationDecision` dataclass, `FrequencyPolicy` (simple frequency + observation ranking), `CostAwareSlackPolicy` (sophisticated utility-aware ranking with reuse probability, recent support/streak, length quality, pruning, memory pressure, PCIe transfer cost).
- **`utility_policy.py`** — `UtilityPolicyController` wrapping the `UtilityModel`, with EWMA hit-rate and waste-ratio feedback.
- **`utility.py`** — `UtilityModel` with `AdmissionEvent`/`UtilityBreakdown`. Computes health-adjusted utility (`U = B - C - W - λ·M - γ·Q`) with semantic uncertainty, divergence penalties, memory opportunity cost, PCIe transfer cost, and template/length/entropy signals.
- **`utility_admission.py`** — `OnlineUtilityEstimator` with bucket-based EWMA cost tracking for fast admission decisions.
- **`controller.py`** — `AdaptiveReuseController` compatibility wrapper around `UtilityPolicyController`.

### Semantic Layer (`semantic.py`)
- **`TokenSketcher`** — Lightweight dependency-free feature hashing using unigram+bigram hash functions for approximate similarity.
- **`SemanticKVIndex`** — In-memory semantic index with sketch caching and query result caching. Supports equivalence key boosting for paraphrase workloads.
- **`token_entropy()`** — Entropy calculation over token distribution.
- **`longest_common_prefix_len()`** — Simple LCP function.

### Backend / Model Layer (`models.py`)
- **`Backend`** — Abstract base with `tokenize()`, `decode()`, `prefill()`, `logit_guard_distance()`, `move_kv_cache()`, `compress_kv_cache()`, `slice_past_key_values()`.
- **`FakeBackend`** — In-process deterministic backend for testing. Supports compression, async prefill, and Jaccard-based logit guard distance.
- **`HuggingFaceBackend`** — Wraps `transformers` models with DynamicCache, GPU utilization tracking (pynvml), logit guard (top-k distribution comparison), KV cache slicing, position-aware prefill.
- **`VLLMBackend`** — Wraps vLLM `LLM` for native prefix caching. No external KV passthrough; primarily a baseline.
- **`HybridBackend`** — Composite backend with primary/secondary fallback.
- **`PrefillResult`** — Dataclass for prefill output (KV cache, latency, memory, utilization, fallback reason).
- **`load_backend()`** — Factory function.
- **`estimate_past_key_values_bytes()`** — Memory estimation utility.
- GPU utilization helpers via pynvml.

### Backend Adapters (`backend_adapters.py`)
- **`BackendAdapter`** ABC and implementations (`HuggingFaceAdapter`, `FakeBackendAdapter`, `GenericBackendAdapter`). Currently NOT wired into engines — marked EXPERIMENTAL.

### Config Layer (`config_loader.py`, `config_watcher.py`)
- **`RuntimeConfig`** — Thread-safe process-wide config singleton with mtime/hash-based hot reloading, dotted-path access, caching, and YAML load/dump.
- Config file: `config/config.yaml` — Hardware params (beta prefill ms, reuse overhead, memory bandwidth, PCIe bandwidth), utility policy params, semantic index params, telemetry settings, raw gate params.

### Safety / Guard Layer
- **`backend/fake_backend.py`** — `SemanticSafetySandbox` validating semantic KV candidates before reuse (logit guard or token-set Jaccard divergence check).
- **`prefix_gate.py`** — `breakeven_prefix_len()` and `transfer_ms_per_token()` based on hardware config.
- **`semantic.py`** — `SemanticSafetyResult` dataclass.

### Other Files
- **`energy.py`** — Likely energy-aware utility metrics (requires inspection).
- **`rl_policy.py`**, **`policy_learning.py`** — RL-based policy learning variants.
- **`telemetry.py`** — Telemetry/decision logging.
- **`workload.py`**, **`datasets.py`** — Workload and dataset construction utilities.
- **`utils.py`** — General utilities.
- **`metrics.py`** — `RunSummary` dataclass and `summarize_run()` function for aggregating engine results.

## Important Files and Directories

| Path | Description |
|------|-------------|
| `src/proactive_kv_cache/` | Core package |
| `src/proactive_kv_cache/cache.py` | `TieredStateBank` — tiered KV cache store |
| `src/proactive_kv_cache/engines.py` | All caching engines (28K+ lines) |
| `src/proactive_kv_cache/policy.py` | Speculation policies (Frequency, CostAwareSlack) |
| `src/proactive_kv_cache/base_policy.py` | Policy base classes and `ReusePlan` |
| `src/proactive_kv_cache/utility.py` | Utility model for admission decisions |
| `src/proactive_kv_cache/utility_policy.py` | UtilityPolicyController |
| `src/proactive_kv_cache/utility_admission.py` | Online utility estimator |
| `src/proactive_kv_cache/controller.py` | AdaptiveReuseController |
| `src/proactive_kv_cache/semantic.py` | Semantic KV index and token sketcher |
| `src/proactive_kv_cache/models.py` | Backends (Fake, HF, vLLM, Hybrid) |
| `src/proactive_kv_cache/backend_adapters.py` | (EXPERIMENTAL) adapter system |
| `src/proactive_kv_cache/backend/fake_backend.py` | Semantic safety sandbox |
| `src/proactive_kv_cache/config_loader.py` | RuntimeConfig with hot reload |
| `src/proactive_kv_cache/prefix_gate.py` | Breakeven prefix length calculation |
| `src/proactive_kv_cache/metrics.py` | RunSummary and summarize_run |
| `config/config.yaml` | Main configuration file |
| `tests/` | 18 test files (conftest, smoke, cache, policy, engine, semantic, overhaul, publishable, literature adapters, etc.) |
| `experiments/` | 14 experiment scripts including run_benchmark, semantic ablation matrix, GPU/CPU sweep scripts |
| `profiling/` | HW profiler (`hw_profiler.py`) and saved profiles |
| `docs/` | 16 research/design documents, reports, LaTeX paper |
| `benchmark_semantic_optimization.py` | Standalone benchmark for semantic analysis optimization |
| `literature_accurate_baselines/` | External runtime baseline adapters and profiles |
| `_inspect_*/` | Archive inspection directories from earlier experiments |

## Build, Test, and Run Commands

**Installation** (from pyproject.toml):
```bash
pip install -e .
```
Dependencies: numpy, matplotlib, transformers, torch, accelerate, safetensors, sentencepiece, huggingface-hub, pydantic, pandas, datasets, pynvml.

**Tests:**
```bash
pytest -q                              # Quick smoke
pytest tests/                          # Full test suite
pytest -k "smoke or fake"              # Filter by keyword
```

**Benchmark (fake backend):**
```bash
python experiments/run_benchmark.py --backend fake --workload synthetic --variant high_skew --n_requests 10 --include_experimental --disable_arrival_simulation --output_dir tmp_smoke
```

**Benchmark with semantic ablations:**
```bash
python experiments/run_semantic_ablation_matrix.py
```

**CPU/GPU benchmarks:**
```bash
bash experiments/run_cpu_benchmark.sh
bash experiments/run_gpu_benchmark.sh
```

## Data Flow and Integrations

1. **Request Flow**: `serve_tokens()` → observe_request (frequency tracking) → peek_match (trie/prefix lookup) → policy plan (admission decision) → prefill_with_cache_fallback or prefill_full → store reactive/scaffold prefix → record metrics.

2. **Speculation Loop**: Background thread periodically calls policy.rank() to select candidates, then prefill() and store() speculative entries during idle GPU time.

3. **Tiered Cache**: Entries stored in GPU or CPU tier; demotion to disk supported. Promotion when hit count exceeds threshold. Eviction via utility-score ranking.

4. **Semantic Index**: Token-sketcher converts prefix tokens to feature vectors; added on store/bypass. Queried during `_plan_for_request` to find semantically similar cached prefixes.

5. **Admission Control**: UtilityModel evaluates benefit/cost/waste with health adjustment, return-vs-risk tradeoff, semantic uncertainty penalties.

6. **Telemetry**: Optional JSONL decision logging with per-request trace rows.

7. **External Runtimes**: Literature-accurate baselines run external vLLM/SGLang/LMCache/KVFlow servers, separate from in-process engine measurements.

## Conventions and Patterns

- **Naming**: `snake_case` for modules, functions, methods. Classes are `PascalCase`.
- **Typing**: Heavy use of `from __future__ import annotations` and PEP 604 union syntax (`X | None`).
- **Threading**: `threading.RLock` for bank operations; daemon threads for speculation workers.
- **Config**: Dotted-path access via `RuntimeConfig.get('section.key', default)`.
- **Caching**: LRU caches, memoization (`@lru_cache`), sketch caches, query result caches with bounded size and periodic cleanup.
- **Dataclasses**: Extensive use for data transfer objects (CacheEntry, RequestResult, ReusePlan, UtilityBreakdown, etc.).
- **Testing**: FakeBackend for deterministic tests; pytest conventions.
- **Metrics**: Engine metrics stored in dict (`engine_metrics`); aggregated via `snapshot_metrics()` and `summarize_run()`.
- **Research Ablations**: Engine variants controlled by `semantic_ablation_mode`, `raw_strategy` parameters; semantic modes: `safe`, `scaffold_only`, `early_layer`, `logit_guard`, `best_latency`, `raw_observer`.

## Recent Agent Updates
- 2026-06-27T03:07:38.108Z — Comprehensive initial codebase scan completed. Full project structure, architecture, and conventions documented after examining all source directories, tests, experiments, config, and documentation.
- 2026-06-27 — Initial comprehensive codebase scan completed. Full project structure documented.
