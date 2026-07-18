# Gemma 4 Memory-Bound Trace

This folder contains the Gemma 4 Blackwell memory-bound trace copied from `C:\shadowkv\v10\results\memory_bound_trace_gemma4`.

## Coverage

- Models: `google/gemma-4-12B-it`, `google/gemma-4-31B-it`.
- Backend/GPU: vLLM on RTX PRO 6000 Blackwell.
- Seeds: `42`, `123`, `456`, `789`, `999`.
- Engines: `no_cache`, native vLLM APC, and MeritKV.
- Trace shape: 40 fill requests, 30 churn/victim requests, and 30 recovery requests.

## Included Files

- `MEMORY_BOUND_RESULTS.md`: paper-facing summary table.
- `gemma4_memory_bound_summary.csv`: compact aggregate table for quick inspection.
- `gemma_4_12b/` and `gemma_4_31b/`: aggregate JSON plus per-seed summary and trace JSON.
- `MANIFEST_SHA256.txt`: checksums for copied files.
- `SOURCE.txt`: source provenance.

## Interpretation

Use Phase 3 recovery, victim misses/evictions, and declined Phase 1 admissions as the main evidence. Some seed-level generic `hit_rate` values are derived and can exceed 1.0 slightly, so they should not be treated as literal hit fractions.
