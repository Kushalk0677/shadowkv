# ShadowKV++ repository evaluation and fix report

## Problem found

ShadowKV++ was making the correct high-level admission decisions, but the cold-start storage path was inefficient. On a first templated/scaffold request, the engine performed the full request prefill and then performed a second prefill of the shared prefix just to materialize the cache entry. This double-prefill made ShadowKV++ slower than simpler baselines even when the policy selected the right prefix.

This showed up as non-zero store latency in structured workloads and made the main engine lose to simpler cache baselines despite zero wasted compute.

## Fix implemented

Changed `src/proactive_kv_cache/engines.py` so ShadowKV++ can retain a reusable prefix directly from the already-computed full-request KV cache.

New internal methods:

- `_slice_prefill_kv_to_prefix(...)`
- `_store_prefix_from_prefill(...)`
- `_store_scaffold_bypass_prefix_from_prefill(...)`
- `_store_semantic_scaffold_prefix_from_prefill(...)`
- `_store_reactive_prefix_from_prefill(...)`

The serving path now uses these opportunistic prefill-store methods instead of recomputing prefixes after a full prefill.

## Why this makes ShadowKV++ stronger

The fix preserves the intended ShadowKV++ policy behaviour:

- raw workloads still avoid unsafe/noisy reuse;
- templated and scaffolded workloads still get aggressive reusable-prefix materialization;
- semantic/scaffold prompts can be retained as candidates without paying an extra prefill;
- wasted compute stays at zero for the main path;
- the cold-start penalty is removed for structured workloads.

## Validation

Regression tests run:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. pytest -q -p no:cacheprovider \
  tests/test_shadowkv_plus.py \
  tests/test_engine_regressions.py \
  tests/test_overhaul_features.py \
  tests/test_cache.py \
  tests/test_policy.py \
  tests/test_backend_regressions.py \
  tests/test_publishable_features.py \
  tests/test_semantic_sandbox.py
```

Result:

```text
50 passed, 1 skipped
```

The skipped test is the optional slow Hugging Face correctness check gated by `RUN_HF_KV_CORRECTNESS=1`.

## Benchmark sanity check after fix

Synthetic high-skew templated workload, fake backend, 64 requests, arrival simulation disabled:

| Engine | Mean latency ms | Speedup vs no-cache | Hit rate | Waste ratio | Opportunistic stores | Store latency ms |
|---|---:|---:|---:|---:|---:|---:|
| no_cache | 34.94 | 1.000x | 0.000 | 0.000 | - | 0.00 |
| reactive_prefix_cache | 22.34 | 1.564x | 0.922 | 0.000 | - | 63.75 |
| shadow_kv | 23.86 | 1.465x | 0.781 | 0.000 | - | 40.75 |
| shadow_kv_plus | 21.34 | 1.637x | 0.922 | 0.000 | 5 | 0.00 |
| shadow_kv_plus_best_latency | 21.34 | 1.637x | 0.922 | 0.000 | 5 | 0.00 |

After the fix, main `shadow_kv_plus` is the best performer in this structured benchmark while preserving zero waste.

Synthetic semantic/paraphrase workload, fake backend, 32 requests:

| Engine | Mean latency ms | Speedup vs no-cache | Hit rate | Waste ratio | Semantic opportunities | Opportunistic stores |
|---|---:|---:|---:|---:|---:|---:|
| no_cache | 65.72 | 1.000x | 0.000 | 0.000 | - | - |
| shadow_kv_plus | 58.61 | 1.121x | 0.219 | 0.000 | 24 | 1 |
| shadow_kv_plus_early_layer | 51.11 | 1.286x | 0.875 | 0.000 | 24 | 1 |

## Remaining caveat

This fix improves ShadowKV++ by removing avoidable implementation overhead. It does not change the correctness boundary: real backends should still avoid unsafe approximate semantic KV reuse unless guarded by an explicit safety mechanism such as the existing logit-guard or sandbox path.
