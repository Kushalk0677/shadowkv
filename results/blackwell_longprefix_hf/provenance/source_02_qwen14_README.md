# Deliverable Contents

- `REPORT_FOR_KUSHAL.md`: results, interpretation, anomalies, and limitations.
- `aggregate_summary.csv`: engine mean, minimum, and maximum metrics.
- `dataset_results.csv`: paired dataset-level results.
- `anomaly_audit.json`: machine-readable completeness and reuse-path checks.
- `analyze_results.py`: reproducible aggregation and audit script.
- `raw_results/`: full-sweep JSON results, manifests, policy traces, per-cell CSVs, logs, and hardware/runtime metadata.
- `smoke_results/`: 16-request compatibility and reuse-path validation.
- `run_logs/`: top-level sweep, status, and nohup logs.
- `source_snapshot/`: exact patched benchmark source used for the run.

Run `analyze_results.py` with Python and pandas to regenerate the summary and audit files.

