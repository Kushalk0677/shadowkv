from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.hw_detect import apply_detected_config

DEFAULT_RESULTS_ROOT = ROOT / "results_6000_qwen14b"

MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEVICE = "cuda"
DTYPE = "float16"
N_REQUESTS = 256
MEAN_INTER_ARRIVAL_MS = "50"
MAX_ARRIVAL_SLEEP_MS = "500"
IDLE_BASELINE_SECONDS = "10"
SEEDS = [42]

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

PROMPT_MODES = ["raw", "templated", "semantic", "rag"]

# Keep paper-grade HF measurements isolated: one engine per Python process.
ENGINES = ["no_cache", "shadow_kv_plus"]


def model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def configure_logging(results_root: Path) -> logging.Logger:
    results_root.mkdir(parents=True, exist_ok=True)
    logs_root = results_root / "_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    log_file = results_root / "_sweep.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    return logging.getLogger("runall_for_6000")


def output_dir_for(results_root: Path, prompt_mode: str, seed: int, dataset: str, engine: str) -> Path:
    return results_root / model_tag(MODEL) / prompt_mode / f"seed_{seed}" / dataset / engine


def log_prefix_for(results_root: Path, prompt_mode: str, seed: int, dataset: str, engine: str) -> Path:
    return results_root / "_logs" / model_tag(MODEL) / prompt_mode / f"seed_{seed}" / dataset / engine


def latest_benchmark_json(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    files = sorted(output_dir.glob("benchmark_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def is_completed(output_dir: Path, engine: str) -> bool:
    path = latest_benchmark_json(output_dir)
    if path is None:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data.get(engine), dict) and "mean_latency_ms" in data[engine]


def build_cmd(output_dir: Path, dataset: str, prompt_mode: str, seed: int, engine: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "experiments/run_benchmark.py",
        "--backend",
        "hf",
        "--model",
        MODEL,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--workload",
        "public_dataset",
        "--dataset",
        dataset,
        "--prompt_mode",
        prompt_mode,
        "--n_requests",
        str(args.n_requests),
        "--mean_inter_arrival_ms",
        str(args.mean_inter_arrival_ms),
        "--max_arrival_sleep_ms",
        str(args.max_arrival_sleep_ms),
        "--seed",
        str(seed),
        "--engines",
        engine,
        "--measure_energy",
        "--idle_baseline_seconds",
        str(args.idle_baseline_seconds),
        "--output_dir",
        str(output_dir),
    ]
    if args.enable_policy_tuning and engine == "shadow_kv_plus":
        cmd.extend(
            [
                "--enable_policy_tuning",
                "--policy_tuning_requests",
                str(args.policy_tuning_requests),
                "--policy_tuning_presets",
                "balanced",
                "aggressive_prefix",
                "aggressive_gpu",
            ]
        )
    if args.enable_policy_trace and engine == "shadow_kv_plus":
        cmd.append("--enable_policy_trace")
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    return cmd


def run_cmd(cmd: list[str], log_prefix: Path, logger: logging.Logger, dry_run: bool) -> tuple[int | str, float]:
    logger.info("CMD  " + " ".join(cmd))
    if dry_run:
        return "dry_run", 0.0
    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.time() - start
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(log_prefix) + ".stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    Path(str(log_prefix) + ".stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode == 0:
        logger.info(f"     OK  elapsed={elapsed:.1f}s")
    else:
        logger.error(f"     FAILED rc={proc.returncode}  elapsed={elapsed:.1f}s")
        logger.error((proc.stderr or proc.stdout or "")[-3000:])
    return proc.returncode, elapsed


def iter_jobs(args: argparse.Namespace):
    for seed in args.seeds:
        for prompt_mode in args.prompt_modes:
            for dataset in args.datasets:
                for engine in args.engines:
                    yield {
                        "seed": seed,
                        "prompt_mode": prompt_mode,
                        "dataset": dataset,
                        "engine": engine,
                    }


def import_result_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
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
            row = {
                "model": config.get("model"),
                "dataset": config.get("dataset"),
                "prompt_mode": config.get("resolved_prompt_mode") or config.get("prompt_mode"),
                "seed": config.get("seed"),
                "engine": engine,
                "backend": config.get("backend"),
                "device": config.get("device"),
                "dtype": config.get("dtype"),
                "n_requests": config.get("n_requests"),
                "mean_latency_ms": metrics.get("mean_latency_ms"),
                "p50_latency_ms": metrics.get("p50_latency_ms"),
                "p95_latency_ms": metrics.get("p95_latency_ms"),
                "p99_latency_ms": metrics.get("p99_latency_ms"),
                "throughput_rps": metrics.get("throughput_rps"),
                "service_throughput_rps": metrics.get("service_throughput_rps"),
                "hit_rate": metrics.get("hit_rate"),
                "reuse_attempts": metrics.get("reuse_attempts"),
                "reuse_successes": metrics.get("reuse_successes"),
                "bypassed_matches": metrics.get("bypassed_matches"),
                "reused_prefix_tokens_total": metrics.get("reused_prefix_tokens_total"),
                "recompute_tokens_total": metrics.get("recompute_tokens_total"),
                "backend_reuse_latency_total_ms": metrics.get("backend_reuse_latency_total_ms"),
                "policy_plans_total": metrics.get("policy_plans_total"),
                "policy_bypass_total": metrics.get("policy_bypass_total"),
                "policy_exact_total": metrics.get("policy_exact_total"),
                "policy_semantic_partial_total": metrics.get("policy_semantic_partial_total"),
                "semantic_queries_total": metrics.get("semantic_queries_total"),
                "semantic_matches_total": metrics.get("semantic_matches_total"),
                "wasted_compute_ratio": metrics.get("wasted_compute_ratio"),
                "gpu_energy_j": metrics.get("gpu_energy_j"),
                "gpu_joules_per_request": metrics.get("gpu_joules_per_request"),
                "idle_adjusted_gpu_energy_j": metrics.get("idle_adjusted_gpu_energy_j"),
                "idle_adjusted_joules_per_request": metrics.get("idle_adjusted_joules_per_request"),
                "gpu_joules_per_total_token": metrics.get("gpu_joules_per_total_token"),
                "energy_source": metrics.get("energy_source"),
                "source_json": str(path),
            }
            rows.append(row)
    return rows


def _as_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_comparison_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row.get("model"), row.get("dataset"), row.get("prompt_mode"), row.get("seed"))
        grouped.setdefault(key, {})[str(row.get("engine"))] = row

    comparisons: list[dict] = []
    for (model, dataset, prompt_mode, seed), engines in sorted(grouped.items()):
        baseline = engines.get("no_cache")
        if not baseline:
            continue
        base_mean = _as_float(baseline.get("mean_latency_ms"))
        base_p95 = _as_float(baseline.get("p95_latency_ms"))
        base_energy = _as_float(baseline.get("gpu_joules_per_request"))
        base_idle_energy = _as_float(baseline.get("idle_adjusted_joules_per_request"))
        for engine, row in sorted(engines.items()):
            if engine == "no_cache":
                continue
            mean = _as_float(row.get("mean_latency_ms"))
            p95 = _as_float(row.get("p95_latency_ms"))
            energy = _as_float(row.get("gpu_joules_per_request"))
            idle_energy = _as_float(row.get("idle_adjusted_joules_per_request"))
            comparisons.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "prompt_mode": prompt_mode,
                    "seed": seed,
                    "engine": engine,
                    "no_cache_mean_latency_ms": base_mean,
                    "engine_mean_latency_ms": mean,
                    "speedup_vs_no_cache_mean": None if not base_mean or not mean else base_mean / max(mean, 1e-9),
                    "no_cache_p95_latency_ms": base_p95,
                    "engine_p95_latency_ms": p95,
                    "speedup_vs_no_cache_p95": None if not base_p95 or not p95 else base_p95 / max(p95, 1e-9),
                    "hit_rate": row.get("hit_rate"),
                    "reuse_successes": row.get("reuse_successes"),
                    "reused_prefix_tokens_total": row.get("reused_prefix_tokens_total"),
                    "no_cache_gpu_joules_per_request": base_energy,
                    "engine_gpu_joules_per_request": energy,
                    "energy_reduction_vs_no_cache_pct": None
                    if base_energy is None or energy is None or base_energy <= 0
                    else 100.0 * (base_energy - energy) / base_energy,
                    "no_cache_idle_adjusted_joules_per_request": base_idle_energy,
                    "engine_idle_adjusted_joules_per_request": idle_energy,
                    "idle_adjusted_energy_reduction_vs_no_cache_pct": None
                    if base_idle_energy is None or idle_energy is None or base_idle_energy <= 0
                    else 100.0 * (base_idle_energy - idle_energy) / base_idle_energy,
                    "source_json": row.get("source_json"),
                }
            )
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen 14B RTX 6000 sweep: one engine per process, all datasets/modes.")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--n_requests", type=int, default=N_REQUESTS)
    parser.add_argument("--mean_inter_arrival_ms", default=MEAN_INTER_ARRIVAL_MS)
    parser.add_argument("--max_arrival_sleep_ms", default=MAX_ARRIVAL_SLEEP_MS)
    parser.add_argument("--idle_baseline_seconds", default=IDLE_BASELINE_SECONDS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--prompt_modes", nargs="+", default=PROMPT_MODES)
    parser.add_argument("--engines", nargs="+", default=ENGINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_policy_tuning", action="store_true")
    parser.add_argument("--policy_tuning_requests", type=int, default=24)
    parser.add_argument("--enable_policy_trace", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    logger = configure_logging(results_root)

    # Same automatic hardware config preflight as run_all_baselines_sweep.py.
    # This updates config.yaml with detected GPU memory and PCIe bandwidth before
    # any benchmark subprocesses are launched. Dry runs avoid mutating config.
    detected_config = {} if args.dry_run else apply_detected_config(log=True)

    jobs = list(iter_jobs(args))
    failed: list[dict] = []
    job_results: list[dict] = []

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "device": args.device,
        "dtype": args.dtype,
        "n_requests": args.n_requests,
        "datasets": args.datasets,
        "prompt_modes": args.prompt_modes,
        "engines": args.engines,
        "seeds": args.seeds,
        "measure_energy": True,
        "idle_baseline_seconds": args.idle_baseline_seconds,
        "one_engine_per_process": True,
        "auto_detected_config": detected_config,
        "total_jobs": len(jobs),
    }
    (results_root / "_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("=" * 80)
    logger.info(
        f"RUNALL_FOR_6000 START model={MODEL} n={args.n_requests} "
        f"datasets={len(args.datasets)} modes={len(args.prompt_modes)} engines={args.engines} jobs={len(jobs)}"
    )
    logger.info("=" * 80)

    start_all = time.time()
    for index, job in enumerate(jobs, start=1):
        output_dir = output_dir_for(results_root, job["prompt_mode"], job["seed"], job["dataset"], job["engine"])
        log_prefix = log_prefix_for(results_root, job["prompt_mode"], job["seed"], job["dataset"], job["engine"])
        logger.info("")
        logger.info(
            f"JOB {index}/{len(jobs)} dataset={job['dataset']} mode={job['prompt_mode']} "
            f"engine={job['engine']} seed={job['seed']}"
        )

        if args.skip_completed and is_completed(output_dir, job["engine"]):
            logger.info("     SKIP completed")
            job_results.append({**job, "returncode": "skipped_completed", "elapsed_sec": 0.0, "output_dir": str(output_dir)})
            continue

        cmd = build_cmd(output_dir, job["dataset"], job["prompt_mode"], job["seed"], job["engine"], args)
        returncode, elapsed = run_cmd(cmd, log_prefix, logger, args.dry_run)
        result = {**job, "returncode": returncode, "elapsed_sec": round(elapsed, 3), "output_dir": str(output_dir)}
        job_results.append(result)
        if returncode != 0 and returncode != "dry_run":
            failed.append(result)
            if not args.continue_on_failure:
                break

        (results_root / "_job_results.json").write_text(json.dumps(job_results, indent=2), encoding="utf-8")

    rows = import_result_rows(results_root)
    comparisons = build_comparison_rows(rows)
    write_csv(results_root / "all_results.csv", rows)
    write_csv(results_root / "comparisons_vs_no_cache.csv", comparisons)
    (results_root / "_job_results.json").write_text(json.dumps(job_results, indent=2), encoding="utf-8")

    elapsed_all = time.time() - start_all
    logger.info("=" * 80)
    logger.info(f"DONE elapsed={elapsed_all:.1f}s failed={len(failed)}/{len(jobs)}")
    logger.info(f"Saved {results_root / 'all_results.csv'}")
    logger.info(f"Saved {results_root / 'comparisons_vs_no_cache.csv'}")
    logger.info("=" * 80)
    return 1 if failed and not args.continue_on_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
