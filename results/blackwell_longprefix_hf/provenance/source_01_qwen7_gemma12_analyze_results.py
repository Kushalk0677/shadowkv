#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "raw_results"


def values_for_key(value, key):
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for child in value.values():
            yield from values_for_key(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from values_for_key(child, key)


def main() -> None:
    all_results = pd.read_csv(RESULTS / "all_results.csv")
    comparisons = pd.read_csv(RESULTS / "comparisons_vs_no_cache.csv")
    reuse = pd.read_csv(RESULTS / "reuse_path_breakdown.csv")
    result_jsons = list(RESULTS.glob("**/benchmark_*.json"))
    fallback_values = []
    for result_json in result_jsons:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        fallback_values.extend(values_for_key(payload, "reuse_backend_fallbacks"))

    metric_columns = [
        "speedup_vs_no_cache_mean",
        "speedup_vs_no_cache_p95",
        "energy_reduction_vs_no_cache_pct",
        "idle_adjusted_energy_reduction_vs_no_cache_pct",
        "hit_rate",
        "reuse_successes",
        "reused_prefix_tokens_total",
        "wasted_compute_ratio",
    ]
    aggregate = (
        comparisons.groupby(["model", "engine"])[metric_columns]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in aggregate.columns
    ]
    aggregate.to_csv(ROOT / "aggregate_summary.csv", index=False)

    dataset_columns = [
        "model",
        "dataset",
        "engine",
        *metric_columns,
    ]
    comparisons[dataset_columns].to_csv(ROOT / "dataset_results.csv", index=False)

    failure_columns = [
        column
        for column in all_results.columns
        if any(token in column.lower() for token in ("fail", "fallback", "error"))
    ]
    numeric_failures = {}
    for column in failure_columns:
        values = pd.to_numeric(all_results[column], errors="coerce")
        if values.notna().any():
            numeric_failures[column] = float(values.fillna(0).sum())

    plus = comparisons[comparisons["engine"] == "shadow_kv_plus"]
    native = comparisons[comparisons["engine"] == "shadow_kv"]
    expected_path = reuse[reuse["engine"] == "shadow_kv_plus"]
    audit = {
        "row_counts": {
            "all_results": len(all_results),
            "comparisons": len(comparisons),
            "reuse_path_breakdown": len(reuse),
        },
        "expected_counts": {
            "all_results": 60,
            "comparisons": 40,
            "reuse_path_breakdown": 40,
        },
        "missing_comparison_metrics": {
            column: int(comparisons[column].isna().sum()) for column in metric_columns
        },
        "numeric_failure_totals": numeric_failures,
        "result_json_count": len(result_jsons),
        "reuse_backend_fallbacks_total": sum(
            value for value in fallback_values if isinstance(value, (int, float))
        ),
        "shadowkv_plus": {
            "cells": len(plus),
            "hit_rate_values": sorted(plus["hit_rate"].dropna().unique().tolist()),
            "reuse_success_values": sorted(
                plus["reuse_successes"].dropna().unique().tolist()
            ),
            "reused_token_values": sorted(
                plus["reused_prefix_tokens_total"].dropna().unique().tolist()
            ),
            "waste_ratio_values": sorted(
                plus["wasted_compute_ratio"].dropna().unique().tolist()
            ),
            "path_values": sorted(expected_path["path_reading"].dropna().unique().tolist()),
            "cells_below_mean_parity": int(
                (plus["speedup_vs_no_cache_mean"] < 1.0).sum()
            ),
            "cells_below_p95_parity": int(
                (plus["speedup_vs_no_cache_p95"] < 1.0).sum()
            ),
        },
        "shadowkv_native": {
            "cells": len(native),
            "nonzero_reuse_cells": int((native["reuse_successes"] > 0).sum()),
        },
    }
    audit["checks"] = {
        "all_expected_rows_present": audit["row_counts"] == audit["expected_counts"],
        "shadowkv_plus_all_cells_exact_scaffold_only": audit["shadowkv_plus"][
            "path_values"
        ]
        == ["exact_scaffold_only"],
        "shadowkv_plus_all_cells_127_reuses": audit["shadowkv_plus"][
            "reuse_success_values"
        ]
        == [127],
        "shadowkv_plus_all_cells_zero_waste": audit["shadowkv_plus"][
            "waste_ratio_values"
        ]
        == [0.0],
        "native_shadowkv_no_reuse": audit["shadowkv_native"][
            "nonzero_reuse_cells"
        ]
        == 0,
        "no_numeric_failures": all(value == 0 for value in numeric_failures.values()),
        "all_result_jsons_present": len(result_jsons) == 60,
        "zero_reuse_backend_fallbacks": audit["reuse_backend_fallbacks_total"] == 0,
    }
    (ROOT / "anomaly_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(aggregate.to_string(index=False))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
