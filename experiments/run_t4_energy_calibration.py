#!/usr/bin/env python3
"""
run_t4_energy_calibration.py — Small energy measurement experiment for T4 GPU.

Purpose:
  Measure actual GPU energy (via NVML) for a representative subset of the
  controlled benchmark matrix. The measured J/req values are then used by
  calibrate_energy_from_measurement.py to estimate energy for the full
  898-file T4/P100 result bundle.

Design:
  1 model  ×  2 datasets  ×  2 modes  ×  1 seed  ×  9 engines = 36 cells
  ~2 min/cell  →  ~72 min total (or ~40 min with 5-engine subset)

  Every cell runs with --measure_energy so the output JSON contains:
    gpu_energy_j
    idle_adjusted_gpu_energy_j
    gpu_joules_per_request
    avg_power_w_from_energy
    energy_reduction_vs_no_cache_pct

Output:
  results/t4_energy_calibration/
    _run_manifest.json
    all_results.csv
    <model>/<mode>/seed_<seed>/<dataset>/<engine>/benchmark_*.json

Usage:
  # Full 36-cell run
  python experiments/run_t4_energy_calibration.py

  # Quick 20-cell smoke test (no frequency_speculative, no ablations)
  python experiments/run_t4_energy_calibration.py --quick
"""

import argparse, subprocess, sys, os, json, time, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASETS = ["ag_news", "samsum"]
MODES = ["templated", "semantic"]
SEEDS = [42]
N_REQUESTS = 256

# All main engines including semantic ablations
ALL_ENGINES = [
    "no_cache",
    "reactive_prefix_cache",
    "greedy_prefix_cache",
    "strict_reactive_prefix_cache",
    "frequency_speculative",
    "shadow_kv",
    "shadow_kv_plus",
    "shadow_kv_plus_best_latency",
    "shadow_kv_plus_raw_observer",
]

# Quick subset — skip frequency_speculative and best-latency/raw-observer variants
QUICK_ENGINES = [
    "no_cache",
    "reactive_prefix_cache",
    "greedy_prefix_cache",
    "shadow_kv",
    "shadow_kv_plus",
]


def make_benchmark_cmd(dataset, mode, seed, engine, output_dir, quick=False):
    """Build the benchmark command for one cell."""
    cmd = [
        sys.executable, str(REPO / "experiments" / "run_benchmark.py"),
        "--backend", "hf",
        "--model", MODEL,
        "--device", "cuda",
        "--dtype", "float16",
        "--workload", "public_dataset",
        "--dataset", dataset,
        "--prompt_mode", mode,
        "--n_requests", str(N_REQUESTS),
        "--seed", str(seed),
        "--include_experimental",
        "--measure_energy",
        "--disable_arrival_simulation",
        "--output_dir", str(output_dir),
    ]
    if engine != "all":
        cmd.extend(["--engines", engine])
    if mode == "semantic":
        cmd.append("--include_semantic_ablations")
    return cmd


def main():
    p = argparse.ArgumentParser(description="T4 energy calibration experiment")
    p.add_argument("--quick", action="store_true",
                   help="Run 5-engine subset (~40 min) instead of full 9-engine (~72 min)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running")
    args = p.parse_args()

    engines = QUICK_ENGINES if args.quick else ALL_ENGINES
    label = "quick" if args.quick else "full"
    out_root = REPO / "results" / f"t4_energy_calibration_{label}"

    total_cells = len(DATASETS) * len(MODES) * len(SEEDS) * len(engines)
    print(f"T4 Energy Calibration — {label} matrix")
    print(f"  Model:     {MODEL}")
    print(f"  Datasets:  {', '.join(DATASETS)}")
    print(f"  Modes:     {', '.join(MODES)}")
    print(f"  Seeds:     {SEEDS}")
    print(f"  Engines:   {', '.join(engines)}")
    print(f"  Cells:     {total_cells}")
    print(f"  Output:    {out_root}")
    print()

    manifest = {
        "experiment": "t4_energy_calibration",
        "model": MODEL,
        "datasets": DATASETS,
        "modes": MODES,
        "seeds": SEEDS,
        "engines": engines,
        "n_requests_per_cell": N_REQUESTS,
        "measure_energy": True,
        "gpu": "NVIDIA T4",
        "cells_total": total_cells,
        "cells_started": 0,
        "cells_completed": 0,
        "cells_failed": 0,
        "started_at": None,
        "completed_at": None,
        "failures": [],
    }

    if args.dry_run:
        print("Dry-run mode — commands that would execute:")
        for dataset in DATASETS:
            for mode in MODES:
                for seed in SEEDS:
                    for engine in engines:
                        cell_dir = out_root / f"{MODEL.replace('/', '_')}" / mode / f"seed_{seed}" / dataset / engine
                        cmd = make_benchmark_cmd(dataset, mode, seed, engine, cell_dir, args.quick)
                        print(f"  {' '.join(cmd)}")
        print(f"\nTotal: {total_cells} cells")
        return

    # Create output root
    out_root.mkdir(parents=True, exist_ok=True)

    manifest["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["cells_started"] = 0
    manifest["cells_completed"] = 0
    manifest["cells_failed"] = 0
    manifest["failures"] = []

    for dataset in DATASETS:
        for mode in MODES:
            for seed in SEEDS:
                for engine in engines:
                    cell_dir = out_root / f"{MODEL.replace('/', '_')}" / mode / f"seed_{seed}" / dataset / engine
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    cmd = make_benchmark_cmd(dataset, mode, seed, engine, cell_dir, args.quick)

                    manifest["cells_started"] += 1
                    _write_manifest(manifest, out_root)

                    print(f"\n[{manifest['cells_started']}/{total_cells}] "
                          f"{dataset}/{mode}/seed_{seed}/{engine}")

                    try:
                        result = subprocess.run(
                            cmd,
                            cwd=str(REPO),
                            capture_output=True, text=True, timeout=7200
                        )
                        if result.returncode == 0:
                            manifest["cells_completed"] += 1
                            print(f"  OK (exit 0)")
                        else:
                            manifest["cells_failed"] += 1
                            manifest["failures"].append({
                                "cell": f"{dataset}/{mode}/seed_{seed}/{engine}",
                                "returncode": result.returncode,
                                "stderr": result.stderr[-500:],
                            })
                            print(f"  FAILED (exit {result.returncode})")
                            print(f"  stderr: {result.stderr[-300:]}")
                    except subprocess.TimeoutExpired:
                        manifest["cells_failed"] += 1
                        manifest["failures"].append({
                            "cell": f"{dataset}/{mode}/seed_{seed}/{engine}",
                            "error": "timeout",
                        })
                        print(f"  TIMEOUT")
                    except Exception as e:
                        manifest["cells_failed"] += 1
                        manifest["failures"].append({
                            "cell": f"{dataset}/{mode}/seed_{seed}/{engine}",
                            "error": str(e),
                        })
                        print(f"  EXCEPTION: {e}")

                    _write_manifest(manifest, out_root)

    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_manifest(manifest, out_root)

    print(f"\n{'='*60}")
    print(f"Experiment complete.")
    print(f"  Total cells:  {total_cells}")
    print(f"  Completed:    {manifest['cells_completed']}")
    print(f"  Failed:       {manifest['cells_failed']}")
    print(f"  Output:       {out_root}")
    if manifest["failures"]:
        print(f"  Failures:")
        for f in manifest["failures"]:
            print(f"    - {f['cell']}: {f.get('error', f.get('returncode', 'unknown'))}")
    print(f"{'='*60}")


def _write_manifest(manifest, out_root):
    with open(out_root / "_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
