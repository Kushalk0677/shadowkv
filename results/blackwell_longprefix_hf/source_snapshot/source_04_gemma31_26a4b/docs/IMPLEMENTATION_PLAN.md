# ShadowKV++ Implementation Plan

## Overview
This document outlines the current state of the ShadowKV++ codebase and the remaining work needed to reach research-quality, publication-ready status. It reflects what actually exists (as of 2026-05-05) and what still needs to be built.

---

## Current State: What Works

### Core Engine (`src/proactive_kv_cache/`)
| Component | Status | Notes |
|---|---|---|
| `cache.py` — `TieredStateBank` | Working | Trie-based prefix matching (reverted from suffix array), GPU/CPU/disk tiers, eviction with utility scoring, LRU/EMA/frequency tracking. All operations serialized under single `threading.RLock()`. |
| `policy.py` — `CostAwareSlackPolicy` | Working | 27 hyperparameters, frequency + recency + streak-based ranking, benefit/cost scoring, dominance pruning. `FrequencyPolicy` simpler baseline also available. |
| `utility.py` — Admission controller | Working | `AdaptiveReuseController` with hit-rate feedback loop, `ReusePlan` decisions |
| `utility_policy.py` / `utility_admission.py` | Working | Online utility estimation and policy-based admission |
| `controller.py` | Working | Reuse planning and feedback integration |
| `engines.py` | Working | 2813 lines. `ReactivePrefixCacheEngine` (baseline), `ShadowKVPlusEngine` (speculative), `ShadowKVPlusLiteEngine` (lightweight). Full integration of trie matching, policy ranking, semantic sandboxing, KV cache slicing, tier migration, and cost-aware admission. |
| `models.py` | Working | `HuggingFaceBackend` (real transformers inference, KV cache management, pynvml GPU utilization), `VLLMBackend` (real vLLM integration, no external KV passthrough), `FakeBackend` (word-level tokenizer, dict KV cache, zlib compression, async prefill), `HybridBackend` (fallback delegation) |
| `prefix_gate.py` | Partial | `transfer_ms_per_token()` and `breakeven_prefix_len()` exist and are called from engines, but depend on `config_loader.CONFIG` (not per-run parameterized) |
| `backend_adapters.py` | Partial | `BackendAdapter` ABC, `HuggingFaceAdapter`, `FakeBackendAdapter`, `GenericBackendAdapter`, `BackendAdapterFactory` exist but are **orphaned** — not wired into any engine |
| `semantic.py` — `SemanticKVIndex` | Configured | Semantically-enhanced cache reuse via embedding-based similarity matching, with long_common_prefix_len and token_entropy utilities. Wired into engines and uses config.yaml settings. |
| `config_loader.py` — `RuntimeConfig` | Working | YAML-based config loading with mtime-watched auto-reload, `watch` method supports live config updates |
| `config_watcher.py` | Working | File system watcher for config.yaml changes, integrates with `RuntimeConfig.reload()` |
| `energy.py` | Working | Energy estimation and tracking via pynvml, used during benchmarks |
| `metrics.py` | Working | `summarize_run()` and `summarize_metrics_report()`, full metric collection across runs |
| `policy_learning.py` / `rl_policy.py` | Partial | PPO-based RL policy training infrastructure with `SimplePPOPolicy` and `train_rl_policy_online()`, but not yet integrated into the main engine loop (experimental) |
| `datasets.py` | Working | `build_benchmark_dataset()` for loading/processing HuggingFace datasets |
| `workload.py` | Working | `BaseWorkload` ABC and `TraceWorkload`, `PublicDatasetWorkload` for trace replay and dataset-based benchmarks |
| `telemetry.py` | Working | Decision logging for post-hoc analysis as per config.yaml |

### Literature-Accurate External Baselines (`literature_accurate_baselines/`)
| Component | Status |
|---|---|
| `adapter_lib.py` (672 lines) | Working. `OpenAICompatClient` (HTTP client with `cached_tokens` extraction), `ManagedServer` (subprocess spawner with health checks), `ExternalAdmissionController` (ShadowKV++ admission overlay via cache resets), workload builders, trace loaders, command builders for vLLM/SGLang/LMCache. |
| `run_runtime_cache_baseline.py` (411 lines) | Working. Unified runner for vLLM APC, SGLang RadixAttention (✅ fixed: added `--enable-cache-report`), and LMCache (✅ fixed: added `--enable-cache-report` to both vLLM and SGLang paths). Supports admission presets and calibration. |
| `run_lmcache.py` (127 lines) | Working. Standalone LMCache runner (transfer or offload mode). |
| `run_kvflow.py` (111 lines) | Working. External adapter for KVFlow. |
| `run_sglang_hicache.py` (124 lines) | Working. SGLang HiCache adapter with full parameter configurability. |
| `oracle_engine.py` (120 lines) | Working. Offline oracle with perfect future knowledge. |
| `run_oracle_future_reuse.py` (105 lines) | Working. Oracle engine runner. |
| `.md` documentation files | Complete for all baselines |

### Testing (`tests/`)
| File | Status |
|---|---|
| `test_cache.py` | Working pytest |
| `test_policy.py` | Working pytest |
| `test_config_utility.py` | Working pytest |
| `test_backend_regressions.py` | Working pytest |
| `test_engine_regressions.py` | Working pytest |
| `test_literature_adapters.py` | Working pytest |
| `test_smoke_fake_backend.py` | Working pytest |
| `test_overhaul_features.py` | Working pytest |
| `test_shadowkv_plus.py` | Working pytest |
| `test_shadowkv_plus_lite.py` | Working pytest |
| `test_semantic_sandbox.py` | Working pytest |
| `test_phase2_phase3.py` | Working pytest |
| `test_publishable_features.py` | Working pytest |
| `test_runtime_baselines.py` | Working pytest |
| `test_chaos_resilience.py` | Script (not pytest — uses `if __name__ == "__main__"`, print-based, no assertions) |
| **Total** | **77 passing**, 1 skipped, 3 pynvml deprecation warnings |

### Experiments (`experiments/`)
| File | Status |
|---|---|
| `run_benchmark.py` | Working. Full benchmark driver with policy tuning, semantic ablations, config injection, workload generation, and multi-backend support (hf, vllm). |
| `run_all_baselines_sweep.py` | Working, tested. Sweeps 11 models × 4 prompt modes × 10 datasets × seeds with runtime baselines and automatic hardware detection. |
| `hw_detect.py` | Working. Auto-detects GPU memory (pynvml → torch → nvidia-smi), PCIe bandwidth (nvidia-smi → lane detection), system RAM (/proc/meminfo), applies values to config.yaml. |
| `render_report.py` | Working. Multi-page research report from benchmark results. |
| `build_policy_dataset_from_traces.py` | Working. Builds PPO training dataset from ShadowKV++ decision traces. |
| `analyze_shadowkv_results.py` | Working. Summarizes cache hit rates, latency savings, and admission metrics from benchmark results. |
| `run_semantic_ablation_matrix.py` | Working. Ablation matrix for semantic vs. raw vs. gated modes. |
| `run_semantic_novelty_matrix.py` | Working. Novelty detection benchmark matrix. |
| `run_cpu_matrix.py` | Working. CPU-only performance benchmark matrix. |
| `smoke_test.py` | Working. Smoke test runner for basic functionality validation. |

---

## Phase 1: Correctness Fixes (Done)
### Goal: Eliminate bugs and config gaps

1. **Add missing `--enable-cache-report` to vLLM + LMCache adapters** ✅
   - Without this flag, vLLM returns `cached_tokens = 0` in API responses.
   - Added to `build_lmcache_command` for both vLLM transfer and offload modes.
   - Verified with full test suite: 77 passing.

2. **Add missing `--enable-cache-report` to SGLang LMCache adapter** ✅
   - Same issue for SGLang + LMCache path.
   - Added alongside `--enable-lmcache` flag.
   - 77 tests still passing.

3. **Add missing `--enable-cache-report` to SGLang RadixAttention baseline** ✅
   - `build_sglang_radix_attention_command` was missing this flag.
   - Added alongside `--enable-metrics`.
   - Fixes the RadixAttention shadowkv_plus admission controller receiving no calibratable cache-hint data.
   - 77 tests still passing.

4. **Auto-detect and apply hardware config at sweep start** ✅
   - `experiments/hw_detect.py` detects GPU memory, PCIe bandwidth, system RAM.
   - `apply_detected_config(log=True)` wired into `run_all_baselines_sweep.py` main().
   - Writes values into config.yaml so cache budgets and PCIe estimates match the actual machine.
   - Verified 13 config keys added to `config/config.yaml`.
   - 77 tests still passing.

5. **Add missing config keys to config.yaml** ✅
   - Added `pcie_bandwidth_gbps`, `max_gpu_memory_mb`, `max_cpu_memory_mb`, `min_bootstrap_admissions`, `lambda_m_weight`, `gamma_q_weight`, and 7 `policy.lite.*` keys to close the gap between documented and actual tunable parameters.
   - 77 tests still passing.

### Status: ✅ Complete. All 77 tests passing, zero regressions.

---

## Phase 2: Bug Fixes and Cleanup (1-2 Weeks)
### Goal: Fix known bugs and orphaned code

1. **Fix dead code in `cache.py:promote_from_disk` (line 130-131)**
   - Second `with self._lock:` references undefined `name` and `value` variables → `UnboundLocalError`.
   - Fix: Remove the dead code block.

2. **Fix `_remove_unlocked` cache invalidation**
   - Currently calls `self._find_match_unlocked.cache_clear()` on every removal — expensive and defeats the cache.
   - Fix: Use targeted `.cache_pop()` or cache only under explicit conditions.

3. **Remove duplicate `_read_gpu_utilization` in `models.py`**
   - Defined twice (lines 648 and 729). The second silently overwrites the first.
   - Fix: Consolidate into a single definition.

4. **Wire `BackendAdapter` system into engines**
   - The `BackendAdapter` ABC and `HuggingFaceAdapter`/`FakeBackendAdapter`/`GenericBackendAdapter` exist but are never called by any engine.
   - Decision: Either wire them in (replace direct backend calls with adapter delegation) or mark as experimental and document why they're orphaned.

5. **Convert chaos test to pytest**
   - `tests/test_chaos_resilience.py` uses `if __name__ == "__main__"` with print statements and no assertions.
   - Fix: Add proper pytest fixtures, assertions for success rate and latency bounds, parameterize chaos modes.

6. **Remove unused `VLLMBackend` dead code**
   - The sweep never uses the inline `VLLMBackend` in `models.py` — it uses the external HTTP adapter via `run_runtime_cache_baseline.py`.
   - `VLLMBackend.prefill()` raises `RuntimeError` if `past_key_values` is provided. This is correct for the external adapter path (which doesn't pass KV), but the class is confusingly present.
   - Decision: Keep but clearly mark as `@deprecated` with inline documentation explaining that the external adapter path is the correct benchmark path.

---

## Phase 3: Wire Up Existing Components (2 Weeks)
### Goal: Connect pieces that exist but aren't integrated

1. **Wire `prefix_gate.py` into `CostAwareSlackPolicy`**
   - `breakeven_prefix_len()` and `transfer_ms_per_token()` exist and produce correct analytical values.
   - Currently used in engines but depends on global `config_loader.CONFIG`.
   - Parametrize for per-run config injection.

2. **Wire `RL Policy` into the engine**
   - `policy_learning.py` has `SimplePPOPolicy` and `train_rl_policy_online()`.
   - `experiments/build_policy_dataset_from_traces.py` can build the training dataset from decision traces.
   - Currently experimental — not part of the main engine loop.
   - Wire as an optional policy mode (e.g., `--policy rl`) with a flag to enable/disable during benchmarks.

3. **Wire `BackendAdapter` into `engines.py`**
   - Use the adapter layer instead of direct backend calls for KV cache preprocessing/postprocessing, benefit estimation, and cache reuse checks.
   - Enables clean vLLM/SGLang adapter additions without modifying engine core.

---

## Phase 4: Performance and Scalability (3-4 Weeks — Optional)
### Goal: Address bottlenecks for production-scale workloads

1. **Fine-grained locking in `TieredStateBank`**
   - Currently a single `threading.RLock()` serializes all operations.
   - Replace with read-write lock: read-only lookups share, writes are exclusive.
   - Add per-tier locks for cross-tier operations (promote/demote).
   - Test: Verify no regression under single-threaded load, measure improvement at 10+ concurrent requests.

2. **Real disk tier implementation**
   - Currently `self.disk_entries` is an in-memory dict — no actual disk I/O.
   - Implement sqlite3 or memory-mapped file storage with serde for KV cache tensors.
   - Add `max_disk_memory_mb` to config.yaml (currently missing).
   - Wire into `demote_to_disk` / `promote_from_disk` methods.

3. **Quantized KV cache for CPU tier**
   - `HuggingFaceBackend.compress_kv_cache()` and `decompress_kv_cache()` exist as stubs.
   - Implement INT8 quantization via `torch.quantization` for CPU-stored prefixes.
   - Lazy decompression on promotion to GPU tier.

4. **Batch speculation during idle periods**
   - `ShadowKVPlusEngine` has speculative precompute, but only precomputes the top-K policy candidates for the current request.
   - Add idle-time batch precompute when GPU utilization drops below threshold.

---

## Phase 5: Features That Don't Matter Yet (Future)
### Goal: Research-grade extensions to explore

1. **Distributed caching layer**
   - Shard `TieredStateBank` across nodes for multi-GPU/multi-node serving.
   - Start with shared LMCache for cross-instance prefix reuse (already available via `run_runtime_cache_baseline.py --baseline lmcache`).

2. **LMCache MP mode adapter**
   - Currently only in-process `LMCacheConnectorV1` is supported.
   - Add `lmcache server` + `LMCacheMPConnector` path for multi-instance sharing and better crash isolation.
   - Note: Not a blocker — in-process mode already exercises the real LMCache system.

3. **Automated hyperparameter tuning**
   - `CostAwareSlackPolicy` has 27 hyperparameters with no calibration procedure.
   - Wire Optuna search over config.yaml values, driven by `run_benchmark.py` as the evaluation harness.
   - Export optimal presets to `config/autotuned.yaml`.

4. **SGLang adapter in `backend_adapters.py`**
   - Add `SGLangAdapter` to the factory alongside `HuggingFaceAdapter`.
   - Enables engine-level integration with SGLang (currently only available via external HTTP adapter).

---

## Current Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Layer                              │
│  experiments/run_benchmark.py                                   │
│  experiments/run_all_baselines_sweep.py                        │
│  experiments/hw_detect.py (auto-detects hardware before sweep) │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Inline       │ │  Literature- │ │  Trace Replay     │
│  Backend      │ │  Accurate    │ │  (oracle, KVFlow) │
│  (Fake/HF)   │ │  External    │ │                    │
│               │ │  HTTP Adapters│ │                    │
│  ShadowKVPlus │ │              │ │                    │
│  Engine       │ │  vLLM APC    │ │                    │
│  Lite         │ │  SGLang radix│ │                    │
│  Reactive     │ │  LMCache     │ │                    │
│               │ │  SGLang HiCache│ │                    │
│               │ │  + ShadowKV++ │ │                    │
│               │ │  admission    │ │                    │
│               │ │  overlay      │ │                    │
└───────────────┘ └──────────────┘ └──────────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Engine Layer                            │
│  engines.py: ShadowKVPlusEngine, ShadowKVPlusLiteEngine        │
│              ReactivePrefixCacheEngine                          │
│  policy.py:  CostAwareSlackPolicy, FrequencyPolicy             │
│              + SimplePPOPolicy (experimental RL)                │
│  utility.py: AdaptiveReuseController with feedback loop        │
│  cache.py:   TieredStateBank with trie matching                │
│              (GPU / CPU / disk tiers, eviction, promotion)      │
│  prefix_gate.py: breakeven_prefix_len (analytical)             │
│  semantic.py: SemanticKVIndex (embedding-based similarity)     │
│  config_loader.py: RuntimeConfig with auto-reload              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Backend Layer                                 │
│  HuggingFaceBackend: Real transformers inference               │
│                        KV cache slicing, compression, pynvml    │
│  VLLMBackend: External vLLM server                             │
│                        (no external KV passthrough)             │
│  FakeBackend: Word tokenizer, dict KV, zlib, async prefill     │
│  HybridBackend: Fallback delegation between two backends        │
│  BackendAdapter: Orphaned (exists but not wired into engines)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Test Status

- **77 passing**, 1 skipped (slow HF correctness check), 3 pynvml deprecation warnings
- All core components covered by pytest
- Chaos test needs conversion from print-based script to proper pytest
- `test_publishable_features.py` validates research-grade features for publication

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Performance regressions during refactoring | Benchmark before/after each change using `run_all_baselines_sweep.py` with a small model subset |
| Lock contention reduction changes correctness | Keep single RLock as default; make read-write lock opt-in via config |
| Disk tier adds latency overhead | Async I/O with configurable timeout; fall back to eviction if disk is slow |
| Real hardware dependencies | Existing test suite uses `FakeBackend` + unit tests. External baseline adapters require vLLM/SGLang installed, which is already the case for users who opt into runtime baselines |

---

## Recent Changes (2026-05-05)

1. Added `--enable-cache-report` to `build_sglang_radix_attention_command` — fixes `cached_tokens = 0` for RadixAttention baseline (adapter_lib.py:417)
2. Added `--enable-cache-report` to vLLM LMCache command builder — fixes `cached_tokens = 0` for LMCache baseline (adapter_lib.py:459)
3. Added `--enable-cache-report` to SGLang LMCache command builder — same fix for SGLang path (adapter_lib.py:493)
4. Wired `hw_detect.py` into `run_all_baselines_sweep.py` — auto-detects hardware config before sweep
5. Added 13 previously missing config keys to `config/config.yaml` (PCIe bandwidth, memory budgets, policy presets)
6. Full pytest suite passes: 77 tests, 0 failures