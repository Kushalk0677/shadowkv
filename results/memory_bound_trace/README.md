# Memory-Bound Trace Results

This folder contains cache-pressure traces for MeritKV. The trace is a three-phase workload: fill reusable prefixes, churn the cache with distractors and victims, then measure recovery.

## Contents

| Path | Contents |
|---|---|
| `MEMORY_BOUND_RESULTS.md` | Combined paper-facing summary for Qwen/Phi and Gemma memory-bound traces. |
| `qwen2.5/` | Qwen2.5 Blackwell memory-bound raw artifacts and summary. |
| `gemma4/` | Gemma 4 Blackwell memory-bound raw artifacts and summary. |

## Scope Notes

- Qwen2.5 and Gemma 4 Blackwell artifacts include raw aggregate and per-seed JSON traces.
- The T4/Phi rows are currently summary-only in this public folder.
- Recovery rate is a cache-hit/recovery metric, not a latency metric.
