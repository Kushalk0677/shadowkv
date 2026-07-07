#!/usr/bin/env python3
"""
calibrate_energy_from_measurement.py

Extrapolate GPU energy measurements from a small T4 calibration run to the
full 898-file controlled result bundle (T4+P100, 5 models, 10 datasets,
3 modes, 3 seeds).

Methodology:
  1. Read the calibration JSONs (from run_t4_energy_calibration.py).
  2. For each (engine, dataset, mode) cell, compute:
       joules_per_request  =  gpu_energy_j / requests_seen
       joules_per_ms       =  joules_per_request / mean_latency_ms
  3. Compute per-engine calibration factors (joules_per_ms) averaged across
     the calibration datasets and modes.
  4. Scan all 898 controlled-result JSONs, compute:
       estimated_energy_j  =  requests_seen × mean_latency_ms × engine_factor
       estimated_waste_energy_j = wasted_compute_ms × engine_factor
  5. Output:
       results/t4_energy_calibration/energy_estimates.csv  (per-cell estimates)
       results/t4_energy_calibration/summary_by_engine_energy.csv  (aggregate)

Usage:
  python experiments/calibrate_energy_from_measurement.py
  python experiments/calibrate_energy_from_measurement.py --calibration-dir results/t4_energy_calibration_full
"""

import argparse, json, glob, os, sys
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Standard column order for the output CSV
ENERGY_COLS = [
    "model_slug", "model_params_B", "dataset", "mode", "engine",
    "seed", "gpu",
    "requests_seen", "mean_latency_ms",
    "measured_gpu_energy_j",           # from calibration (NaN for non-calibration cells)
    "estimated_gpu_energy_j",          # calibrated estimate
    "estimated_waste_energy_j",        # wasted speculative compute energy
    "estimated_inference_energy_j",    # total inference energy
    "est_joules_per_request",          # estimated J/req
    "energy_reduction_vs_no_cache_pct",
    "calibration_source",              # "measured" or "extrapolated"
]


def load_calibration_data(cal_dir):
    """Load calibration JSONs and compute per-engine calibration factors."""
    jsons = glob.glob(os.path.join(cal_dir, "**", "benchmark_*.json"), recursive=True)
    if not jsons:
        print(f"ERROR: No benchmark JSONs found in {cal_dir}")
        sys.exit(1)

    rows = []
    for fpath in jsons:
        with open(fpath) as f:
            data = json.load(f)
        config = data.get("config", {})
        for eng_name, eng_data in data.items():
            if eng_name in ("config", "capabilities"):
                continue
            n = eng_data.get("requests_seen", 0)
            if n == 0:
                continue
            lat = eng_data.get("mean_latency_ms", 0)
            energy = eng_data.get("gpu_energy_j", None)
            if energy is None or energy == 0:
                print(f"  WARNING: {os.path.basename(fpath)}/{eng_name} has no gpu_energy_j")
                continue
            wasted_ms = eng_data.get("wasted_compute_ms", 0)
            rows.append({
                "engine": eng_name,
                "dataset": config.get("dataset", "unknown"),
                "mode": config.get("prompt_mode", "unknown"),
                "seed": config.get("seed", 0),
                "requests_seen": n,
                "mean_latency_ms": lat,
                "gpu_energy_j": energy,
                "wasted_compute_ms": wasted_ms,
                "joules_per_request": energy / n,
                "joules_per_ms": energy / (n * lat) if (n * lat) > 0 else 0,
            })

    cal_df = pd.DataFrame(rows)
    if len(cal_df) == 0:
        print("ERROR: No usable calibration data found.")
        sys.exit(1)

    # Compute per-engine calibration factors (mean joules_per_ms across calibration cells)
    engine_factors = cal_df.groupby("engine").agg(
        mean_joules_per_ms=("joules_per_ms", "mean"),
        std_joules_per_ms=("joules_per_ms", "std"),
        n_cal_cells=("joules_per_ms", "count"),
    ).reset_index()

    print(f"Loaded {len(rows)} calibration cells from {len(jsons)} JSONs")
    print(f"Calibration factors computed for {len(engine_factors)} engines")
    print()
    print(f"  {'Engine':35s} {'J/ms':>10s} {'±':>5s} {'Cells':>6s}")
    print(f"  {'-'*56}")
    for _, r in engine_factors.iterrows():
        print(f"  {r['engine']:35s} {r['mean_joules_per_ms']:8.4f} ±{r['std_joules_per_ms']:6.4f} {r['n_cal_cells']:6d}")
    print()

    return cal_df, engine_factors


def estimate_all_results(cal_df, engine_factors, result_root):
    """Scan all controlled-result JSONs and estimate energy for each cell."""
    files = glob.glob(os.path.join(result_root, "**", "benchmark_*.json"), recursive=True)
    if not files:
        print(f"ERROR: No benchmark JSONs found in {result_root}")
        sys.exit(1)

    # Build engine factor lookup
    factor_map = dict(zip(engine_factors["engine"], engine_factors["mean_joules_per_ms"]))

    # For engines not in calibration, use the nearest match
    ENGINE_FALLBACK = {
        "shadow_kv_plus_best_latency": "shadow_kv_plus",
        "shadow_kv_plus_raw_observer": "shadow_kv_plus",
        "shadow_kv_plus_scaffold_only": "shadow_kv_plus",
        "shadow_kv_plus_early_layer": "shadow_kv_plus",
        "shadow_kv_plus_logit_guard": "shadow_kv_plus",
    }

    rows = []
    missing_engines = set()

    for fpath in files:
        # Extract gpu type from path (t4 or p100)
        path_parts = fpath.replace("\\", "/").split("/")
        gpu = "t4" if "t4" in path_parts else ("p100" if "p100" in path_parts else "unknown")

        with open(fpath) as f:
            data = json.load(f)
        config = data.get("config", {})

        for eng_name, eng_data in data.items():
            if eng_name in ("config", "capabilities"):
                continue
            n = eng_data.get("requests_seen", 0)
            if n == 0:
                continue
            lat = eng_data.get("mean_latency_ms", 0)
            wasted_ms = eng_data.get("wasted_compute_ms", 0)

            # Get calibration factor
            factor = factor_map.get(eng_name)
            if factor is None:
                fallback = ENGINE_FALLBACK.get(eng_name)
                if fallback:
                    factor = factor_map.get(fallback)
                if factor is None:
                    if eng_name not in missing_engines:
                        missing_engines.add(eng_name)
                    continue

            est_energy = n * lat * factor
            est_waste = wasted_ms * factor

            # Find matching calibration cell
            cal_match = cal_df[
                (cal_df["engine"] == eng_name) &
                (cal_df["dataset"] == config.get("dataset")) &
                (cal_df["mode"] == config.get("prompt_mode"))
            ]
            measured_energy = cal_match["gpu_energy_j"].values[0] if len(cal_match) > 0 else None

            rows.append({
                "model_slug": config.get("model", "unknown").replace("/", "_"),
                "model_params_B": _get_params_B(config.get("model", "")),
                "dataset": config.get("dataset", "unknown"),
                "mode": config.get("prompt_mode", "unknown"),
                "engine": eng_name,
                "seed": config.get("seed", 0),
                "gpu": gpu,
                "requests_seen": n,
                "mean_latency_ms": lat,
                "measured_gpu_energy_j": measured_energy if measured_energy else None,
                "estimated_gpu_energy_j": est_energy,
                "estimated_waste_energy_j": est_waste,
                "estimated_inference_energy_j": est_energy,
                "est_joules_per_request": est_energy / n if n > 0 else 0,
                "energy_reduction_vs_no_cache_pct": None,  # computed in aggregate
                "calibration_source": "measured" if measured_energy else "extrapolated",
            })

    if missing_engines:
        print(f"WARNING: No calibration factor for engines: {sorted(missing_engines)}")

    out_df = pd.DataFrame(rows)

    # Compute energy_reduction_vs_no_cache_pct per (model, dataset, mode, seed, gpu)
    no_cache = out_df[out_df["engine"] == "no_cache"][
        ["model_slug", "dataset", "mode", "seed", "gpu", "estimated_gpu_energy_j"]
    ].rename(columns={"estimated_gpu_energy_j": "nc_energy"})

    out_df = out_df.merge(no_cache, on=["model_slug", "dataset", "mode", "seed", "gpu"], how="left")
    out_df["energy_reduction_vs_no_cache_pct"] = (
        (out_df["nc_energy"] - out_df["estimated_gpu_energy_j"]) / out_df["nc_energy"] * 100
    )
    out_df.drop(columns=["nc_energy"], inplace=True)

    return out_df


def _get_params_B(model_name):
    """Return approximate parameter count in billions."""
    model_name = (model_name or "").lower()
    if "32b" in model_name and "1.5" not in model_name:
        return 32.5
    if "14b" in model_name:
        return 14.7
    if "7b" in model_name:
        return 7.61
    if "3b" in model_name:
        return 3.09
    if "1.5b" in model_name:
        return 1.54
    if "phi-3" in model_name:
        return 3.8
    if "gemma-2b" in model_name or "gemma-2b" in model_name:
        return 2.0
    if "tinyllama" in model_name or "llama" in model_name:
        return 1.1
    if "gpt2" in model_name:
        return 0.124
    return 0


def main():
    p = argparse.ArgumentParser(description="Calibrate GPU energy from T4 measurement")
    p.add_argument("--calibration-dir", default=str(REPO / "results" / "t4_energy_calibration_full"),
                   help="Directory containing calibration benchmark JSONs")
    p.add_argument("--result-root", default=str(REPO / "results" / "controlled_results"),
                   help="Root of controlled result bundle")
    p.add_argument("--output-dir", default=str(REPO / "results" / "t4_energy_calibration"),
                   help="Output directory for energy estimates")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("T4 Energy Calibration — Extrapolating Measured Energy to Full Bundle")
    print("=" * 60)

    print(f"\n[1] Loading calibration data from {args.calibration_dir}")
    cal_df, engine_factors = load_calibration_data(args.calibration_dir)

    print(f"[2] Scanning controlled results in {args.result_root}")
    est_df = estimate_all_results(cal_df, engine_factors, args.result_root)
    print(f"  {len(est_df)} cells estimated ({est_df['calibration_source'].value_counts().to_dict()})")

    # Save full estimates
    est_path = out_dir / "energy_estimates_full.csv"
    est_df[ENERGY_COLS].to_csv(est_path, index=False)
    print(f"\n[3] Full estimates saved to {est_path}")

    # Aggregate by engine
    print(f"\n[4] Aggregating by engine...")
    agg = est_df.groupby("engine").agg(
        total_energy_kj=("estimated_gpu_energy_j", "sum"),
        total_waste_energy_kj=("estimated_waste_energy_j", "sum"),
        total_requests=("requests_seen", "sum"),
        mean_joules_per_request=("est_joules_per_request", "mean"),
        mean_energy_reduction_pct=("energy_reduction_vs_no_cache_pct", "mean"),
        n_cells=("estimated_gpu_energy_j", "count"),
    ).reset_index()
    agg["total_energy_kj"] /= 1000
    agg["total_waste_energy_kj"] /= 1000

    # Separate T4 and P100
    for gpu in ["t4", "p100"]:
        gpu_df = est_df[est_df["gpu"] == gpu]
        if len(gpu_df) == 0:
            continue
        gpu_agg = gpu_df.groupby("engine").agg(
            total_energy_kj=("estimated_gpu_energy_j", "sum"),
            total_waste_energy_kj=("estimated_waste_energy_j", "sum"),
            total_requests=("requests_seen", "sum"),
            mean_joules_per_request=("est_joules_per_request", "mean"),
            mean_energy_reduction_pct=("energy_reduction_vs_no_cache_pct", "mean"),
            n_cells=("estimated_gpu_energy_j", "count"),
        ).reset_index()
        gpu_agg["total_energy_kj"] /= 1000
        gpu_agg["total_waste_energy_kj"] /= 1000
        gpu_agg_path = out_dir / f"summary_by_engine_energy_{gpu}.csv"
        gpu_agg.to_csv(gpu_agg_path, index=False)
        print(f"  {gpu.upper()} summary: {gpu_agg_path}")

    agg_path = out_dir / "summary_by_engine_energy.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  Combined summary: {agg_path}")

    # Print headline table
    print()
    print("=" * 60)
    print("Headline Energy Estimates (T4 + P100 combined)")
    print("=" * 60)
    print(f"  {'Engine':35s} {'Total KJ':>9s} {'Waste KJ':>9s} {'J/req':>7s} {'vs NC':>7s} {'Cells':>6s}")
    print(f"  {'-'*73}")
    for _, r in agg.iterrows():
        print(f"  {r['engine']:35s} {r['total_energy_kj']:>8.1f}K {r['total_waste_energy_kj']:>8.1f}K "
              f"{r['mean_joules_per_request']:>6.2f} {r['mean_energy_reduction_pct']:>+5.1f}% {r['n_cells']:>6d}")

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
