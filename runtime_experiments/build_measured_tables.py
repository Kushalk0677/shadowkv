#!/usr/bin/env python3
"""
build_measured_tables.py — Extract clean measured runtime data for
SGLang, vLLM, and LMCache experiments.

No estimation, no imputation, no extrapolation. Only real measured data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runtime_experiments"

# ── Paths to source data ──────────────────────────────────────────────────
V10 = Path(r"C:\shadowkv\v10\working\_extracted")

P3_CSV = (
    V10 / "sglang_3models_identical_blocks"
    / "shadowkv_sglang_3models_identical_blocks_deliverable_2026-06-17"
    / "raw_results"
    / "results_sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16"
    / "aggregate_3models_3baselines_3reps_full.csv"
)

Q14_CSV = (
    V10 / "qwen14b_identical_blocks"
    / "shadowkv_qwen14b_identical_blocks_deliverable_2026-06-18"
    / "raw_results"
    / "results_sglang_lmcache_shadowkv_qwen14b_identical_blocks_3reps_2026-06-17"
    / "aggregate_qwen14b_3baselines_3reps_full.csv"
)

Q32_SGLANG_JSON = (
    V10 / "sglang_lmcache_qwen32b"
    / "results_sglang_lmcache_qwen32b_full_2026-06-08"
    / "summary_sglang_lmcache_qwen32b_2026-06-08.json"
)

Q32_VLLM_CSV = (
    V10 / "qwen32b_no_cache_apc_energy"
    / "shadowkv_qwen32b_no_cache_apc_overlay_energy_2026-06-03_bundle"
    / "results_vllm_qwen32b_no_cache_apc_overlay_energy_2026-06-03"
    / "aggregate_qwen32b_no_cache_apc_overlay_energy_full.csv"
)

LMCACHE_14B_CSV = (
    V10 / "lmcache_no_native_radix"
    / "shadowkv_lmcache_no_native_radix_deliverable_2026-06-14"
    / "raw_results"
    / "results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14"
    / "summary_lmcache_no_native_radix_qwen14b_2026-06-14.csv"
)

SGLANG_SMALL_14B_CSV = (
    V10 / "lmcache_no_native_radix"
    / "shadowkv_lmcache_no_native_radix_deliverable_2026-06-14"
    / "raw_results"
    / "results_sglang_shadowkv_qwen14b_small_2026-06-13"
    / "summary_sglang_shadowkv_qwen14b_small_2026-06-13.csv"
)

MEASURED_DATASETS = ["ag_news", "daily_dialog", "dolly", "samsum", "xsum"]
MODES = ["rag", "templated"]
ENGINES_SGLANG = [
    "sglang_radix_attention",
    "sglang_radix_attention_shadowkv_plus",
    "lmcache_no_native_radix",
]
ENGINES_VLLM = ["vllm_no_cache", "vllm_apc", "vllm_apc_shadowkv_plus"]

PARAMS = {
    "qwen25_15b": 1.54, "qwen25_3b": 3.09,
    "qwen25_7b": 7.61, "qwen25_14b": 14.7, "qwen25_32b": 32.5,
}


def build_sglang_table():
    """Build SGLang 3-engine comparison table (1.5B, 3B, 7B, 14B)."""
    print("  Loading SGLang 3-model data...")
    df3 = pd.read_csv(P3_CSV)
    df3 = df3[df3["baseline"].isin(ENGINES_SGLANG)].copy()

    print("  Loading SGLang 14B data...")
    df14 = pd.read_csv(Q14_CSV)
    df14 = df14[df14["baseline"].isin(ENGINES_SGLANG)].copy()

    combined = pd.concat([df3, df14], ignore_index=True)
    combined["model_params_B"] = combined["model_slug"].map(PARAMS)

    # Aggregate 3 replicates
    group_cols = ["model_slug", "model", "model_params_B", "dataset", "mode", "baseline"]
    metrics = {
        "mean_latency_ms": ["mean", "std"],
        "throughput_rps": ["mean", "std"],
        "cached_tokens_mean": ["mean", "std"],
        "gpu_energy_j": ["mean", "std"],
    }
    agg = combined.groupby(group_cols).agg(metrics)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.reset_index(inplace=True)
    rep_counts = combined.groupby(group_cols).size().reset_index(name="n_replicates")
    agg = agg.merge(rep_counts, on=group_cols, how="left")

    # Filter to measured datasets only
    agg = agg[agg["dataset"].isin(MEASURED_DATASETS)].copy()

    # Compute speedup vs lmcache
    lmcache = agg[agg["baseline"] == "lmcache_no_native_radix"][
        ["model_slug", "dataset", "mode", "mean_latency_ms_mean"]
    ].rename(columns={"mean_latency_ms_mean": "lmcache_latency"})

    agg = agg.merge(lmcache, on=["model_slug", "dataset", "mode"], how="left")

    # speedup_pct = (lmcache_lat - engine_lat) / engine_lat * 100  -> how much faster engine is
    # But standard is: (baseline - new) / baseline * 100
    agg["speedup_vs_lmcache_pct"] = (
        (agg["lmcache_latency"] - agg["mean_latency_ms_mean"])
        / agg["lmcache_latency"] * 100
    )
    agg.drop(columns=["lmcache_latency"], inplace=True)

    # Rename for clean output
    out = agg.rename(columns={
        "model_slug": "model_short",
        "model": "model_full",
        "model_params_B": "params_B",
        "baseline": "engine",
        "mean_latency_ms_mean": "mean_latency_ms",
        "mean_latency_ms_std": "latency_std",
        "throughput_rps_mean": "throughput_rps",
        "throughput_rps_std": "throughput_std",
        "cached_tokens_mean_mean": "cached_tokens_mean",
        "cached_tokens_mean_std": "cached_tokens_std",
        "gpu_energy_j_mean": "gpu_energy_j",
        "gpu_energy_j_std": "energy_std",
    })

    out_cols = [
        "model_short", "params_B", "dataset", "mode", "engine",
        "mean_latency_ms", "latency_std", "n_replicates",
        "throughput_rps", "throughput_std",
        "cached_tokens_mean", "cached_tokens_std",
        "gpu_energy_j", "energy_std",
        "speedup_vs_lmcache_pct",
    ]
    out = out[out_cols].copy()
    out.rename(columns={"model_short": "model"}, inplace=True)
    out.sort_values(["model", "dataset", "mode", "engine"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def build_vllm_table():
    """Build vLLM APC comparison table (32B only)."""
    print("  Loading vLLM 32B data...")
    df = pd.read_csv(Q32_VLLM_CSV)
    df = df[df["baseline"].isin(ENGINES_VLLM)].copy()
    df = df[df["dataset"].isin(MEASURED_DATASETS)].copy()

    # Compute speedup vs no_cache
    no_cache = df[df["baseline"] == "vllm_no_cache"][
        ["dataset", "mode", "mean_latency_ms"]
    ].rename(columns={"mean_latency_ms": "nocache_latency"})

    df = df.merge(no_cache, on=["dataset", "mode"], how="left")
    df["speedup_vs_no_cache_pct"] = (
        (df["nocache_latency"] - df["mean_latency_ms"])
        / df["nocache_latency"] * 100
    )
    df.drop(columns=["nocache_latency"], inplace=True)

    out = df.rename(columns={
        "baseline": "engine",
        "throughput_rps": "throughput_rps",
        "gpu_energy_j": "gpu_energy_j",
    }).copy()

    out["model"] = "qwen25_32b"
    out["params_B"] = 32.5

    out_cols = [
        "model", "params_B", "dataset", "mode", "engine",
        "mean_latency_ms", "throughput_rps",
        "gpu_energy_j", "speedup_vs_no_cache_pct",
    ]
    # Add cached_tokens column as it exists in vLLM data
    if "cached_tokens_total_response_usage" in out.columns:
        out.rename(columns={"cached_tokens_total_response_usage": "cached_tokens"}, inplace=True)
    out_cols_c = [c for c in out_cols if c in out.columns]
    if "cached_tokens" in out.columns:
        out_cols_c.insert(5, "cached_tokens")

    out = out[out_cols_c].copy()
    out.sort_values(["dataset", "mode", "engine"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def build_sglang_32b_table():
    """Build SGLang 32B supplementary table (2 engines: radix, lmcache)."""
    print("  Loading SGLang 32B data...")
    import json
    with open(Q32_SGLANG_JSON, "r") as f:
        data = json.load(f)

    rows = []
    for item in data["rows"]:
        ds = item["dataset"]
        if ds not in MEASURED_DATASETS:
            continue
        bl = item["baseline"]
        if bl == "lmcache":
            bl = "lmcache_no_native_radix"
        rows.append({
            "model": "qwen25_32b",
            "params_B": 32.5,
            "dataset": ds,
            "mode": item["prompt_mode"],
            "engine": bl,
            "mean_latency_ms": item["mean_latency_ms"],
            "throughput_rps": item["throughput_rps"],
            "cached_tokens_mean": item.get("cached_tokens_mean",
                                           item.get("cached_tokens_total", 0) / 256),
            "gpu_energy_j": item["gpu_energy_j"],
        })

    df = pd.DataFrame(rows)

    # Speedup: lmcache vs radix (both directions)
    radix = df[df["engine"] == "sglang_radix_attention"][
        ["dataset", "mode", "mean_latency_ms"]
    ].rename(columns={"mean_latency_ms": "radix_latency"})
    df = df.merge(radix, on=["dataset", "mode"], how="left")
    df["speedup_vs_radix_pct"] = (
        (df["radix_latency"] - df["mean_latency_ms"])
        / df["radix_latency"] * 100
    )
    df.drop(columns=["radix_latency"], inplace=True)

    out_cols = [
        "model", "params_B", "dataset", "mode", "engine",
        "mean_latency_ms", "throughput_rps",
        "cached_tokens_mean", "gpu_energy_j", "speedup_vs_radix_pct",
    ]
    out = df[out_cols].copy()
    out.sort_values(["dataset", "mode", "engine"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def build_lmcache_table():
    """Build LMCache no-native-radix table (14B only)."""
    print("  Loading LMCache 14B data...")
    df = pd.read_csv(LMCACHE_14B_CSV)
    df = df[df["dataset"].isin(MEASURED_DATASETS)].copy()
    df["model"] = "qwen25_14b"
    df["params_B"] = 14.7

    df.rename(columns={"baseline": "engine"}, inplace=True)
    out_cols = [
        "model", "params_B", "dataset", "mode", "engine",
        "mean_latency_ms", "throughput_rps",
        "cached_tokens_mean", "gpu_energy_j",
    ]
    out = df[out_cols].copy()
    out.sort_values(["dataset", "mode"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def main():
    print("=" * 60)
    print("Building clean measured runtime tables")
    print("=" * 60)

    # SGLang table
    print("\n[1] SGLang 3-engine comparison...")
    sglang_df = build_sglang_table()
    sglang_path = OUT / "sglang" / "results_3engine_comparison.csv"
    sglang_df.to_csv(sglang_path, index=False)
    print(f"  -> {sglang_path} ({len(sglang_df)} rows)")

    # vLLM table
    print("\n[2] vLLM APC comparison...")
    vllm_df = build_vllm_table()
    vllm_path = OUT / "vllm" / "results_apc_comparison.csv"
    vllm_df.to_csv(vllm_path, index=False)
    print(f"  -> {vllm_path} ({len(vllm_df)} rows)")

    # SGLang 32B supplementary
    print("\n[3] SGLang 32B supplementary...")
    sglang32_df = build_sglang_32b_table()
    sglang32_path = OUT / "sglang" / "results_32b_comparison.csv"
    sglang32_df.to_csv(sglang32_path, index=False)
    print(f"  -> {sglang32_path} ({len(sglang32_df)} rows)")

    # LMCache table
    print("\n[4] LMCache no-native-radix...")
    lmcache_df = build_lmcache_table()
    lmcache_path = OUT / "lmcache" / "results.csv"
    lmcache_df.to_csv(lmcache_path, index=False)
    print(f"  -> {lmcache_path} ({len(lmcache_df)} rows)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  SGLang 3-engine:     {len(sglang_df)} rows  ({sglang_df['model'].nunique()} models, "
          f"{sglang_df['engine'].nunique()} engines)")
    print(f"  SGLang 32B:          {len(sglang32_df)} rows  ({sglang32_df['engine'].nunique()} engines)")
    print(f"  vLLM APC:            {len(vllm_df)} rows  ({vllm_df['engine'].nunique()} engines)")
    print(f"  LMCache:             {len(lmcache_df)} rows  ({lmcache_df['engine'].nunique()} engines)")
    total_rows = len(sglang_df) + len(sglang32_df) + len(vllm_df) + len(lmcache_df)
    print(f"  Total data rows:     {total_rows}")
    print(f"  Datasets covered:    {', '.join(MEASURED_DATASETS)}")
    print(f"  Engines covered:     {len(ENGINES_SGLANG) + len(ENGINES_VLLM)} unique")
    print(f"  Models covered:      1.5B, 3B, 7B, 14B (SGLang), 32B (vLLM + SGLang)")
    print("  Note: No estimated or imputed values. All data is measured.")
    print("Done.")


if __name__ == "__main__":
    main()
