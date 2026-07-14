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

    comparison_metrics = [
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
        comparisons.groupby(["model", "engine"])[comparison_metrics]
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

    engine_metrics = [
        "mean_latency_ms",
        "p95_latency_ms",
        "throughput_rps",
        "gpu_joules_per_request",
        "idle_adjusted_joules_per_request",
    ]
    engine_summary = (
        all_results.groupby(["model", "engine"])[engine_metrics]
        .mean()
        .reset_index()
    )
    engine_summary.to_csv(ROOT / "engine_summary.csv", index=False)

    comparisons[["model", "dataset", "engine", *comparison_metrics]].to_csv(
        ROOT / "dataset_results.csv", index=False
    )

    plus = comparisons[comparisons["engine"] == "shadow_kv_plus"]
    native = comparisons[comparisons["engine"] == "shadow_kv"]
    expected_path = reuse[reuse["engine"] == "shadow_kv_plus"]

    model_rows = []
    for model, group in plus.groupby("model"):
        model_rows.append(
            {
                "model": model,
                "cells": len(group),
                "mean_speedup_vs_no_cache": group["speedup_vs_no_cache_mean"].mean(),
                "p95_speedup_vs_no_cache": group["speedup_vs_no_cache_p95"].mean(),
                "energy_reduction_vs_no_cache_pct": group[
                    "energy_reduction_vs_no_cache_pct"
                ].mean(),
                "idle_adjusted_energy_reduction_vs_no_cache_pct": group[
                    "idle_adjusted_energy_reduction_vs_no_cache_pct"
                ].mean(),
                "cells_below_mean_parity": int(
                    (group["speedup_vs_no_cache_mean"] < 1.0).sum()
                ),
                "cells_below_p95_parity": int(
                    (group["speedup_vs_no_cache_p95"] < 1.0).sum()
                ),
                "mean_speedup_min": group["speedup_vs_no_cache_mean"].min(),
                "mean_speedup_max": group["speedup_vs_no_cache_mean"].max(),
            }
        )
    model_summary = pd.DataFrame(model_rows).sort_values("model")
    model_summary.to_csv(ROOT / "model_summary.csv", index=False)

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

    expected_models = {
        "gpt2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "google/gemma-4-E2B-it",
        "google/gemma-4-12B-it",
    }
    model_counts = {
        model: int(count)
        for model, count in all_results.groupby("model").size().items()
    }
    audit = {
        "row_counts": {
            "all_results": len(all_results),
            "comparisons": len(comparisons),
            "reuse_path_breakdown": len(reuse),
        },
        "expected_counts": {
            "all_results": 210,
            "comparisons": 140,
            "reuse_path_breakdown": 140,
        },
        "model_counts": model_counts,
        "expected_models": sorted(expected_models),
        "result_json_count": len(result_jsons),
        "missing_comparison_metrics": {
            column: int(comparisons[column].isna().sum())
            for column in comparison_metrics
        },
        "numeric_failure_totals": numeric_failures,
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
            "path_values": sorted(
                expected_path["path_reading"].dropna().unique().tolist()
            ),
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
        "all_models_present_with_30_cells": set(model_counts) == expected_models
        and set(model_counts.values()) == {30},
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
        "no_numeric_failures": all(value == 0 for value in numeric_failures.values()),
        "all_result_jsons_present": len(result_jsons) == 210,
        "zero_reuse_backend_fallbacks": audit["reuse_backend_fallbacks_total"] == 0,
    }
    (ROOT / "anomaly_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(model_summary.to_string(index=False))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
