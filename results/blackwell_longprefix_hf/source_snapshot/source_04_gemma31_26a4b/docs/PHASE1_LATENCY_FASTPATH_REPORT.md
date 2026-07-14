# Phase 1: ShadowKV++ latency fast path

## What changed

This draft introduces a latency-first serving variant named `shadow_kv_plus_lite`.

The Lite engine is deliberately narrower than full ShadowKV++:

- exact-prefix KV reuse only;
- no semantic retrieval in the latency path;
- no background speculative worker;
- no rich policy/controller planning for obvious exact-prefix hits;
- break-even minimum-prefix admission before reuse;
- scaffold/prefix storage from the already-computed full prefill;
- aggregate timing counters only, while per-request policy tracing remains opt-in.

Full `shadow_kv_plus` remains the research/control-plane engine for semantic opportunity tracking, policy traces, and ablations.

## New benchmark engine

Use:

```bash
python experiments/run_benchmark.py \
  --backend fake \
  --workload synthetic \
  --variant high_skew \
  --n_requests 128 \
  --engines no_cache shadow_kv_plus_lite shadow_kv_plus \
  --min_reuse_prefix_tokens 96
```

For fake/local smoke testing, use a smaller threshold:

```bash
python experiments/run_benchmark.py \
  --backend fake \
  --workload synthetic \
  --variant high_skew \
  --n_requests 8 \
  --engines no_cache shadow_kv_plus_lite \
  --min_reuse_prefix_tokens 3
```

## New CLI flag

```text
--min_reuse_prefix_tokens
```

This overrides the ShadowKV++ Lite break-even reuse threshold. If omitted, defaults are backend/device-aware:

- fake backend: existing tuning threshold;
- CUDA backend: `policy.lite.min_reuse_prefix_tokens_cuda`, default 96;
- CPU/HF backend: `policy.lite.min_reuse_prefix_tokens_cpu`, default 128.

## New telemetry counters

The following aggregate counters were added to support latency attribution without per-request tracing overhead:

```text
cache_lookup_calls
cache_lookup_latency_total_ms
policy_planning_calls
policy_planning_latency_total_ms
semantic_query_calls
semantic_query_latency_total_ms
kv_materialization_calls
kv_materialization_latency_total_ms
backend_reuse_calls
backend_reuse_latency_total_ms
small_prefix_bypass_total
net_latency_saved_estimate_ms_total
lite_mode_enabled
lite_min_reuse_prefix_tokens
lite_fast_path_total
lite_policy_fallback_total
lite_exact_bypass_total
lite_store_from_prefill_total
```

## Validation

Repository test result after Phase 1 changes:

```text
72 passed, 1 skipped
```

The skipped test is the optional slow Hugging Face KV correctness test guarded by `RUN_HF_KV_CORRECTNESS=1`.

Smoke benchmark on fake/high_skew, 8 requests, `--min_reuse_prefix_tokens 3`:

```text
no_cache mean latency:              34.94 ms
shadow_kv_plus_lite mean latency:   25.56 ms
mean speedup:                       1.37x
hit rate:                           0.625
policy_planning_calls:              0
semantic_query_calls:               0
lite_fast_path_total:               5
```

This smoke result is not a paper benchmark. It only verifies that the Phase 1 hot path works and avoids the full policy/semantic machinery.

## Interpretation

The Phase 1 change separates two roles:

1. `shadow_kv_plus`: research-grade policy and semantic controller.
2. `shadow_kv_plus_lite`: serving-grade latency fast path.

This gives the project a better chance of winning latency benchmarks because the measured hot path no longer pays semantic-query or policy-planning overhead for simple exact-prefix reuse.

## Next Phase 1.5 recommendation

Run real CUDA/HF benchmarks with:

- `max_tokens = 32/64/128`;
- larger prompts and larger models;
- thresholds 64, 96, 128, 192, 256;
- separate latency and policy-trace runs.

The main success criterion should be net utility, not hit rate:

```text
net saved latency = avoided prefill cost - lookup/reuse/materialization overhead
```
