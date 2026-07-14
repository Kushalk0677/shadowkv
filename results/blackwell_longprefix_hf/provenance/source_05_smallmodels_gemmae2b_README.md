# Deliverable Contents

This package contains the complete seven-model ShadowKV++ HF Blackwell long-prefix experiment.

- `REPORT_FOR_KUSHAL.md`: interpretation, results, caveats, and follow-up.
- `model_summary.csv`: one row per model with ShadowKV++ speedup and energy results.
- `engine_summary.csv`: aggregate latency, throughput, and energy by model and engine.
- `aggregate_summary.csv`: mean/min/max comparison metrics by model and cache engine.
- `dataset_results.csv`: dataset-level comparison rows.
- `anomaly_audit.json`: machine-readable completeness and reuse-path audit.
- `raw_results/`: all 210 raw benchmark cells, aggregate CSVs, policy traces, and metadata.
- `smoke_results/`: six-model and Gemma 4 E2B smoke outputs.
- `run_logs/`: full logs and status files for both execution blocks.
- `source_snapshot/`: benchmark source, tests, operator notes, and launch wrappers.
- `analyze_results.py`: reproducible aggregation and audit script.
- `MANIFEST_SHA256.txt`: SHA256 manifest for every packaged file except the manifest itself.

The original runtime result root is:

`/home/jade_hand/research/shadowkv/hf_blackwell_semantic_n128_longprefix_20260710/results_blackwell_semantic_seven_models_longprefix_n128_20260713`

