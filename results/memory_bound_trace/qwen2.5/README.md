# Qwen2.5 Memory-Bound Trace

This folder contains the Qwen2.5 Blackwell memory-bound trace copied from `C:\shadowkv\v10\results\memory_bound_trace_qwen`.

## Coverage

- Models: `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`.
- Backend/GPU: vLLM on RTX PRO 6000 Blackwell.
- Seeds: `42`, `123`, `456`, `789`, `999`.
- Engines: `no_cache`, native vLLM APC, and MeritKV.
- Trace shape: 40 fill requests, 30 churn/victim requests, and 30 recovery requests.

## Included Files

- `MEMORY_BOUND_RESULTS.md`: standalone Qwen2.5 summary table.
- `qwen2.5_memory_bound_summary.csv`: compact aggregate table for quick inspection.
- `qwen2.5_14b/` and `qwen2.5_32b/`: aggregate JSON plus per-seed summary and trace JSON.
- `MANIFEST_SHA256.txt`: checksums for copied files.
- `SOURCE.txt`: source provenance.

## Interpretation

The main paper-facing metrics are Phase 3 recovery, victim misses/evictions, and declined Phase 1 admissions. Qwen2.5-32B is the stronger Blackwell cache-pressure case: MeritKV recovers substantially more Phase 3 reuse while admitting far fewer Phase 1 prefixes.
