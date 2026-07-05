#!/usr/bin/env python3
"""Isolated P100 HF sweep driver.

This runner is intentionally conservative for a 12 GB P100: each engine cell is
its own process. That costs time, but it prevents one failed model/engine cell
from poisoning the next cell with allocator fragmentation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results_p100_isolated"

MODELS = [
    "gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]

DATASETS = [
    "ag_news", "alpaca_eval", "banking77", "cnn_dailymail", "daily_dialog",
    "dolly", "oasst1", "samsum", "ultrachat", "xsum",
]

PROMPT_MODES = ["raw", "templated", "semantic"]
ENGINES = ["no_cache", "shadow_kv", "shadow_kv_plus"]


def tag(model: str) -> str:
    return model.replace("/", "_").replace(".", "_")


def out_dir(root: Path, model: str, mode: str, seed: int, dataset: str, engine: str) -> Path:
    return root / tag(model) / mode / f"seed_{seed}" / dataset / engine


def is_done(path: Path, engine: str) -> bool:
    files = sorted(path.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if path.exists() else []
    if not files:
        return False
    try:
        data = json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data.get(engine), dict) and "mean_latency_ms" in data[engine]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated P100 HF sweep.")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--prompt_modes", nargs="+", default=PROMPT_MODES)
    parser.add_argument("--engines", nargs="+", default=ENGINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--n_requests", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max_memory_mb", type=int, default=256)
    parser.add_argument("--mean_inter_arrival_ms", type=float, default=50.0)
    parser.add_argument("--max_arrival_sleep_ms", type=float, default=500.0)
    parser.add_argument("--allow_unsafe_semantic_kv_reuse", action="store_true")
    parser.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (model, dataset, mode, seed, engine)
        for model in args.models
        for dataset in args.datasets
        for mode in args.prompt_modes
        for seed in args.seeds
        for engine in args.engines
    ]
    (root / "_run_manifest.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "experiments/run_p100_isolated_sweep.py",
        "models": args.models,
        "datasets": args.datasets,
        "prompt_modes": args.prompt_modes,
        "seeds": args.seeds,
        "engines": args.engines,
        "n_requests": args.n_requests,
        "isolation": "one subprocess per model/dataset/prompt_mode/seed/engine",
    }, indent=2), encoding="utf-8")
    failures = []
    with (root / "_sweep.log").open("a", encoding="utf-8") as log:
        for i, (model, dataset, mode, seed, engine) in enumerate(jobs, start=1):
            out = out_dir(root, model, mode, seed, dataset, engine)
            line = f"[{i}/{len(jobs)}] {model} {dataset} {mode} seed={seed} engine={engine}"
            print(line, flush=True)
            log.write(line + "\n")
            if args.skip_completed and is_done(out, engine):
                log.write("  SKIP completed\n")
                continue
            cmd = [
                args.python, "experiments/run_benchmark.py",
                "--backend", "hf",
                "--model", model,
                "--device", args.device,
                "--dtype", args.dtype,
                "--workload", "public_dataset",
                "--dataset", dataset,
                "--prompt_mode", mode,
                "--n_requests", str(args.n_requests),
                "--simulate_arrivals",
                "--mean_inter_arrival_ms", str(args.mean_inter_arrival_ms),
                "--max_arrival_sleep_ms", str(args.max_arrival_sleep_ms),
                "--max_memory_mb", str(args.max_memory_mb),
                "--seed", str(seed),
                "--engines", engine,
                "--output_dir", str(out),
            ]
            if mode == "semantic" and args.allow_unsafe_semantic_kv_reuse:
                cmd.append("--allow_unsafe_semantic_kv_reuse")
            log.write("  CMD " + " ".join(cmd) + "\n")
            if args.dry_run:
                continue
            proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out.mkdir(parents=True, exist_ok=True)
            (out / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
            (out / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            log.write(f"  rc={proc.returncode}\n")
            if proc.returncode != 0:
                failures.append({"model": model, "dataset": dataset, "mode": mode, "seed": seed, "engine": engine, "returncode": proc.returncode})
                if not args.continue_on_failure:
                    break
    (root / "_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return 1 if failures and not args.continue_on_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
