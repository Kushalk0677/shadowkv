# Performance Fast-Path Update

This update addresses the observed regression in raw-mode sweeps after adding semantic metrics, ablations, and request-level tracing.

## Changes

### 1. Policy trace is now opt-in

`policy_trace.jsonl` is no longer emitted by default. This prevents per-request JSON construction and disk I/O from contaminating latency benchmarks.

Use:

```bash
python experiments/run_benchmark.py ... --enable_policy_trace
```

Without the flag, the benchmark JSON records:

```json
"policy_trace_file": null,
"policy_trace_rows": 0
```

### 2. Semantic index is skipped outside semantic/ablation mode

ShadowKV++ no longer performs semantic-index insertion/query work for raw and templated performance runs unless semantic ablations are explicitly enabled. This keeps raw/templated comparisons focused on exact reuse and policy behavior.

New metric:

```text
semantic_queries_skipped_total
```

### 3. Raw-mode fast bypass

ShadowKV++ now has a low-reuse fast-bypass guard for raw workloads. After a small warm-up window, if the observed reuse density remains low, it bypasses cache matching, semantic matching, and reactive store work for subsequent raw requests.

New metric:

```text
fast_raw_bypass_total
```

This is scoped only to:

- `prompt_mode="raw"`
- default safe ShadowKV++ engine
- not scaffold/early-layer/logit-guard ablations

Templated and semantic modes continue to use their scaffold and semantic paths.

## Expected impact

- Raw mode should avoid the worst controller/semantic overhead regressions.
- Templated mode should be mostly unchanged, except lower overhead from skipped semantic queries.
- Semantic mode should preserve the semantic opportunity metrics and safe HF blocking behavior.
- Policy trace data generation remains available, but must be enabled explicitly.
