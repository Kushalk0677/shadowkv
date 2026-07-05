#!/usr/bin/env python3
"""Isolated Blackwell semantic n=128 HF sweep.

This public runner is the cleaned version of the Blackwell handoff script. It
keeps the run restartable and avoids importing historical transfer packages.
Each benchmark cell runs in its own Python subprocess.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results_blackwell_semantic_n128"

NORMAL_MODELS = [
    "gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]

LARGE_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]

GEMMA4_MODELS = [
    "google/gemma-4-12b-it",
    "google/gemma-4-26b-a4b-it",
]

DATASETS = [
    "ag_news",
    "alpaca_eval",
    "banking77",
    "cnn_dailymail",
    "daily_dialog",
    "dolly",
    "oasst1",
    "samsum",
    "ultrachat",
    "xsum",
]

ENGINES = ["no_cache", "shadow_kv", "shadow_kv_plus"]


def model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def output_dir(results_root: Path, model: str, prompt_mode: str, seed: int, dataset: str, engine: str) -> Path:
    return results_root / model_tag(model) / prompt_mode / f"seed_{seed}" / dataset / engine


def latest_json(path: Path) -> Path | None:
    if not path.exists():
        return None
    files = sorted(path.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def completed(path: Path, engine: str) -> bool:
    latest = latest_json(path)
    if latest is None:
        return False
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data.get(engine), dict) and "mean_latency_ms" in data[engine]


def build_command(args: argparse.Namespace, model: str, dataset: str, prompt_mode: str, seed: int, engine: str, out: Path) -> list[str]:
    cmd = [
        args.python,
        "experiments/run_benchmark.py",
        "--backend", "hf",
        "--model", model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--workload", "public_dataset",
        "--dataset", dataset,
        "--prompt_mode", prompt_mode,
        "--n_requests", str(args.n_requests),
        "--seed", str(seed),
        "--engines", engine,
        "--policy_preset", args.policy_preset,
        "--mean_inter_arrival_ms", str(args.mean_inter_arrival_ms),
        "--max_arrival_sleep_ms", str(args.max_arrival_sleep_ms),
        "--max_memory_mb", str(args.max_memory_mb),
        "--gpu_index", str(args.gpu_index),
        "--idle_baseline_seconds", str(args.idle_baseline_seconds),
        "--output_dir", str(out),
    ]
    if args.measure_energy:
        cmd.append("--measure_energy")
    if args.simulate_arrivals:
        cmd.append("--simulate_arrivals")
    else:
        cmd.append("--disable_arrival_simulation")
    if args.allow_unsafe_semantic_kv_reuse:
        cmd.append("--allow_unsafe_semantic_kv_reuse")
    if args.semantic_index_diagnostics:
        cmd.append("--semantic_index_diagnostics")
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    return cmd


def collect_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("benchmark_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = data.get("config", {})
        for engine, metrics in data.items():
            if engine in {"config", "capabilities", "runtime", "idle_energy_baseline"} or not isinstance(metrics, dict):
                continue
            if "mean_latency_ms" not in metrics:
                continue
            rows.append({
                "model": config.get("resolved_model") or config.get("model"),
                "dataset": config.get("dataset"),
                "prompt_mode": config.get("resolved_prompt_mode") or config.get("prompt_mode"),
                "seed": config.get("seed"),
                "engine": engine,
                "mean_latency_ms": metrics.get("mean_latency_ms"),
                "p95_latency_ms": metrics.get("p95_latency_ms"),
                "throughput_rps": metrics.get("throughput_rps"),
                "hit_rate": metrics.get("hit_rate"),
                "wasted_compute_ratio": metrics.get("wasted_compute_ratio"),
                "semantic_partial_hits": metrics.get("semantic_partial_hits"),
                "semantic_opportunity_plans_total": metrics.get("semantic_opportunity_plans_total"),
                "semantic_blocked_by_backend_total": metrics.get("semantic_blocked_by_backend_total"),
                "gpu_energy_j": metrics.get("gpu_energy_j"),
                "gpu_joules_per_request": metrics.get("gpu_joules_per_request"),
                "source_json": str(path),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated Blackwell semantic n=128 HF sweep.")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--n_requests", type=int, default=128)
    parser.add_argument("--models", nargs="+", default=NORMAL_MODELS + LARGE_MODELS + GEMMA4_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--prompt_modes", nargs="+", default=["semantic"])
    parser.add_argument("--engines", nargs="+", default=ENGINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--policy_preset", default="balanced")
    parser.add_argument("--mean_inter_arrival_ms", type=float, default=50.0)
    parser.add_argument("--max_arrival_sleep_ms", type=float, default=500.0)
    parser.add_argument("--max_memory_mb", type=int, default=512)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--idle_baseline_seconds", type=float, default=5.0)
    parser.add_argument("--measure_energy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simulate_arrivals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_unsafe_semantic_kv_reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--semantic_index_diagnostics", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    log_path = results_root / "_sweep.log"
    jobs = [
        (model, dataset, mode, seed, engine)
        for model in args.models
        for dataset in args.datasets
        for mode in args.prompt_modes
        for seed in args.seeds
        for engine in args.engines
    ]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "experiments/run_blackwell_semantic_n128.py",
        "backend": "hf",
        "models": args.models,
        "datasets": args.datasets,
        "prompt_modes": args.prompt_modes,
        "seeds": args.seeds,
        "engines": args.engines,
        "n_requests": args.n_requests,
        "isolation": "one subprocess per model/dataset/prompt_mode/seed/engine",
    }
    (results_root / "_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    failures: list[dict[str, Any]] = []
    with log_path.open("a", encoding="utf-8") as log:
        for index, (model, dataset, mode, seed, engine) in enumerate(jobs, start=1):
            out = output_dir(results_root, model, mode, seed, dataset, engine)
            prefix = f"[{index}/{len(jobs)}] {model} {dataset} {mode} seed={seed} engine={engine}"
            print(prefix, flush=True)
            log.write(prefix + "\n")
            if args.skip_completed and completed(out, engine):
                log.write("  SKIP completed\n")
                continue
            cmd = build_command(args, model, dataset, mode, seed, engine, out)
            log.write("  CMD " + " ".join(cmd) + "\n")
            if args.dry_run:
                continue
            start = time.time()
            proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elapsed = time.time() - start
            out.mkdir(parents=True, exist_ok=True)
            (out / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
            (out / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            log.write(f"  rc={proc.returncode} elapsed_sec={elapsed:.1f}\n")
            if proc.returncode != 0:
                failures.append({"model": model, "dataset": dataset, "mode": mode, "seed": seed, "engine": engine, "returncode": proc.returncode})
                if not args.continue_on_failure:
                    break
        rows = collect_rows(results_root)
        write_csv(results_root / "all_results.csv", rows)
        (results_root / "_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return 1 if failures and not args.continue_on_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
