#!/usr/bin/env python3
"""Isolated Blackwell semantic-reuse n=128 sweep.

This runner is meant to be handed to another person running the RTX PRO 6000
Blackwell machine. It keeps the benchmark simple and isolated:

* one seed by default: 42
* semantic prompt mode by default
* core paper engines: no_cache, shadow_kv, shadow_kv_plus
* one Python subprocess per model/dataset/prompt-mode/engine cell

The isolation is deliberate. It avoids cross-engine contamination from warmed
kernels, backend state, allocator fragmentation, or lingering KV tensors.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.hw_detect import apply_detected_config


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
    # Verify the exact Hugging Face slugs on the target machine before the full run.
    "google/gemma-4-12B-it",
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

PROMPT_MODES = ["semantic"]

# Core ShadowKV comparison engines for the HF backend.
ENGINES = ["no_cache", "shadow_kv", "shadow_kv_plus"]


def model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def configure_logging(results_root: Path) -> logging.Logger:
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "_logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(results_root / "_sweep.log", mode="a", encoding="utf-8"),
        ],
        force=True,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    return logging.getLogger("blackwell_semantic_n128")


def output_dir_for(results_root: Path, model: str, prompt_mode: str, seed: int, dataset: str, engine: str) -> Path:
    return results_root / model_tag(model) / prompt_mode / f"seed_{seed}" / dataset / engine


def log_prefix_for(results_root: Path, model: str, prompt_mode: str, seed: int, dataset: str, engine: str) -> Path:
    return results_root / "_logs" / model_tag(model) / prompt_mode / f"seed_{seed}" / dataset / engine


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


def iter_jobs(args: argparse.Namespace):
    for model in args.models:
        for seed in args.seeds:
            for prompt_mode in args.prompt_modes:
                for dataset in args.datasets:
                    for engine in args.engines:
                        yield {
                            "model": model,
                            "seed": seed,
                            "prompt_mode": prompt_mode,
                            "dataset": dataset,
                            "engine": engine,
                        }


def build_cmd(output_dir: Path, job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "experiments/run_benchmark.py",
        "--backend",
        "hf",
        "--model",
        job["model"],
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--workload",
        "public_dataset",
        "--dataset",
        job["dataset"],
        "--prompt_mode",
        job["prompt_mode"],
        "--n_requests",
        str(args.n_requests),
        "--mean_inter_arrival_ms",
        str(args.mean_inter_arrival_ms),
        "--max_arrival_sleep_ms",
        str(args.max_arrival_sleep_ms),
        "--max_memory_mb",
        str(args.max_memory_mb),
        "--speculative_k",
        str(args.speculative_k),
        "--idle_threshold_ms",
        str(args.idle_threshold_ms),
        "--seed",
        str(job["seed"]),
        "--engines",
        job["engine"],
        "--policy_preset",
        args.policy_preset,
        "--early_layer_reuse_ratio",
        str(args.early_layer_reuse_ratio),
        "--logit_guard_threshold",
        str(args.logit_guard_threshold),
        "--gpu_index",
        str(args.gpu_index),
        "--idle_baseline_seconds",
        str(args.idle_baseline_seconds),
        "--output_dir",
        str(output_dir),
    ]
    if args.measure_energy:
        cmd.append("--measure_energy")
    if args.simulate_arrivals:
        cmd.append("--simulate_arrivals")
    else:
        cmd.append("--disable_arrival_simulation")
    if args.allow_unsafe_semantic_kv_reuse:
        cmd.append("--allow_unsafe_semantic_kv_reuse")
    if args.enable_policy_trace:
        cmd.append("--enable_policy_trace")
    if args.semantic_index_diagnostics:
        cmd.append("--semantic_index_diagnostics")
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    if args.config_path:
        cmd.extend(["--config_path", args.config_path])
    return cmd


def build_run_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["SHADOWKV_SEMANTIC_LONG_PREFIX_REPEATS"] = str(args.semantic_shared_prefix_repeats)
    env["SHADOWKV_SEMANTIC_LONG_PREFIX_MODE"] = str(args.semantic_shared_prefix_mode)
    return env


def run_cmd(
    cmd: list[str],
    log_prefix: Path,
    logger: logging.Logger,
    dry_run: bool,
    run_env: dict[str, str],
) -> tuple[int | str, float]:
    logger.info("CMD  %s", " ".join(cmd))
    if dry_run:
        return "dry_run", 0.0
    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=run_env)
    elapsed = time.time() - start
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(log_prefix) + ".stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    Path(str(log_prefix) + ".stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode == 0:
        logger.info("     OK elapsed=%.1fs", elapsed)
    else:
        logger.error("     FAILED rc=%s elapsed=%.1fs", proc.returncode, elapsed)
        logger.error((proc.stderr or proc.stdout or "")[-3000:])
    return proc.returncode, elapsed


def import_result_rows(results_root: Path) -> list[dict[str, Any]]:
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
            rows.append(
                {
                    "model": config.get("resolved_model") or config.get("model"),
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
                    "speedup_vs_no_cache_mean": metrics.get("speedup_vs_no_cache_mean"),
                    "speedup_vs_no_cache_p95": metrics.get("speedup_vs_no_cache_p95"),
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
                    "fast_exact_path_hits": metrics.get("fast_exact_path_hits"),
                    "scaffold_bypass_store_successes": metrics.get("scaffold_bypass_store_successes"),
                    "semantic_scaffold_store_successes": metrics.get("semantic_scaffold_store_successes"),
                    "scaffold_only_attempts": metrics.get("scaffold_only_attempts"),
                    "scaffold_only_hits": metrics.get("scaffold_only_hits"),
                    "early_layer_attempts": metrics.get("early_layer_attempts"),
                    "early_layer_hits": metrics.get("early_layer_hits"),
                    "logit_guard_attempts": metrics.get("logit_guard_attempts"),
                    "logit_guard_hits": metrics.get("logit_guard_hits"),
                    "semantic_queries_total": metrics.get("semantic_queries_total"),
                    "semantic_matches_total": metrics.get("semantic_matches_total"),
                    "semantic_partial_hits": metrics.get("semantic_partial_hits"),
                    "semantic_partial_reused_tokens_total": metrics.get("semantic_partial_reused_tokens_total"),
                    "semantic_opportunity_plans_total": metrics.get("semantic_opportunity_plans_total"),
                    "semantic_opportunity_reused_tokens_total": metrics.get("semantic_opportunity_reused_tokens_total"),
                    "semantic_opportunity_estimated_savings_ms": metrics.get("semantic_opportunity_estimated_savings_ms"),
                    "semantic_blocked_by_backend_total": metrics.get("semantic_blocked_by_backend_total"),
                    "semantic_quality_divergence_sum": metrics.get("semantic_quality_divergence_sum"),
                    "semantic_quality_divergence_events": metrics.get("semantic_quality_divergence_events"),
                    "wasted_compute_ratio": metrics.get("wasted_compute_ratio"),
                    "gpu_energy_j": metrics.get("gpu_energy_j"),
                    "gpu_joules_per_request": metrics.get("gpu_joules_per_request"),
                    "idle_adjusted_gpu_energy_j": metrics.get("idle_adjusted_gpu_energy_j"),
                    "idle_adjusted_joules_per_request": metrics.get("idle_adjusted_joules_per_request"),
                    "gpu_joules_per_total_token": metrics.get("gpu_joules_per_total_token"),
                    "energy_source": metrics.get("energy_source"),
                    "source_json": str(path),
                }
            )
    return rows


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any, Any], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("model"), row.get("dataset"), row.get("prompt_mode"), row.get("seed"))
        grouped.setdefault(key, {})[str(row.get("engine"))] = row

    comparisons: list[dict[str, Any]] = []
    for (model, dataset, prompt_mode, seed), engines in sorted(grouped.items()):
        baseline = engines.get("no_cache")
        if not baseline:
            continue
        base_mean = as_float(baseline.get("mean_latency_ms"))
        base_p95 = as_float(baseline.get("p95_latency_ms"))
        base_energy = as_float(baseline.get("gpu_joules_per_request"))
        base_idle_energy = as_float(baseline.get("idle_adjusted_joules_per_request"))
        for engine, row in sorted(engines.items()):
            if engine == "no_cache":
                continue
            mean = as_float(row.get("mean_latency_ms"))
            p95 = as_float(row.get("p95_latency_ms"))
            energy = as_float(row.get("gpu_joules_per_request"))
            idle_energy = as_float(row.get("idle_adjusted_joules_per_request"))
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
                    "wasted_compute_ratio": row.get("wasted_compute_ratio"),
                    "policy_exact_total": row.get("policy_exact_total"),
                    "fast_exact_path_hits": row.get("fast_exact_path_hits"),
                    "policy_semantic_partial_total": row.get("policy_semantic_partial_total"),
                    "semantic_partial_hits": row.get("semantic_partial_hits"),
                    "semantic_partial_reused_tokens_total": row.get("semantic_partial_reused_tokens_total"),
                    "semantic_opportunity_plans_total": row.get("semantic_opportunity_plans_total"),
                    "semantic_blocked_by_backend_total": row.get("semantic_blocked_by_backend_total"),
                    "semantic_quality_divergence_sum": row.get("semantic_quality_divergence_sum"),
                    "semantic_quality_divergence_events": row.get("semantic_quality_divergence_events"),
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


def as_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_reuse_path_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path_rows: list[dict[str, Any]] = []
    for row in rows:
        engine = str(row.get("engine"))
        if engine == "no_cache":
            continue
        exact_total = as_int(row.get("policy_exact_total"))
        exact_scaffold = as_int(row.get("fast_exact_path_hits"))
        semantic_partial_total = as_int(row.get("policy_semantic_partial_total"))
        semantic_partial_hits = as_int(row.get("semantic_partial_hits"))
        reuse_successes = as_int(row.get("reuse_successes"))
        path_rows.append(
            {
                "model": row.get("model"),
                "dataset": row.get("dataset"),
                "prompt_mode": row.get("prompt_mode"),
                "seed": row.get("seed"),
                "engine": engine,
                "n_requests": row.get("n_requests"),
                "hit_rate": row.get("hit_rate"),
                "reuse_successes": reuse_successes,
                "reused_prefix_tokens_total": row.get("reused_prefix_tokens_total"),
                "exact_policy_total": exact_total,
                "exact_scaffold_hits": exact_scaffold,
                "exact_non_scaffold_policy_total": max(exact_total - exact_scaffold, 0),
                "semantic_partial_policy_total": semantic_partial_total,
                "semantic_partial_hits": semantic_partial_hits,
                "semantic_partial_reused_tokens_total": row.get("semantic_partial_reused_tokens_total"),
                "semantic_opportunity_plans_total": row.get("semantic_opportunity_plans_total"),
                "semantic_opportunity_reused_tokens_total": row.get("semantic_opportunity_reused_tokens_total"),
                "semantic_opportunity_estimated_savings_ms": row.get("semantic_opportunity_estimated_savings_ms"),
                "scaffold_bypass_store_successes": row.get("scaffold_bypass_store_successes"),
                "semantic_scaffold_store_successes": row.get("semantic_scaffold_store_successes"),
                "scaffold_only_attempts": row.get("scaffold_only_attempts"),
                "scaffold_only_hits": row.get("scaffold_only_hits"),
                "early_layer_attempts": row.get("early_layer_attempts"),
                "early_layer_hits": row.get("early_layer_hits"),
                "logit_guard_attempts": row.get("logit_guard_attempts"),
                "logit_guard_hits": row.get("logit_guard_hits"),
                "path_reading": (
                    "semantic_partial_executed"
                    if semantic_partial_hits > 0
                    else ("exact_scaffold_only" if exact_scaffold > 0 else "no_reuse_path_executed")
                ),
                "source_json": row.get("source_json"),
            }
        )
    return path_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finalize_results(results_root: Path) -> None:
    rows = import_result_rows(results_root)
    write_csv(results_root / "all_results.csv", rows)
    write_csv(results_root / "comparisons_vs_no_cache.csv", build_comparison_rows(rows))
    write_csv(results_root / "reuse_path_breakdown.csv", build_reuse_path_rows(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated Blackwell n=128 semantic-reuse sweep.")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--n_requests", type=int, default=128)
    parser.add_argument("--mean_inter_arrival_ms", type=float, default=50.0)
    parser.add_argument("--max_arrival_sleep_ms", type=float, default=500.0)
    parser.add_argument("--max_memory_mb", type=int, default=512)
    parser.add_argument("--speculative_k", type=int, default=2)
    parser.add_argument("--idle_threshold_ms", type=float, default=30.0)
    parser.add_argument("--models", nargs="+", default=NORMAL_MODELS + LARGE_MODELS + GEMMA4_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--prompt_modes", nargs="+", default=PROMPT_MODES)
    parser.add_argument("--engines", nargs="+", default=ENGINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--policy_preset", default="balanced")
    parser.add_argument("--measure_energy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--idle_baseline_seconds", type=float, default=5.0)
    parser.add_argument("--simulate_arrivals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_unsafe_semantic_kv_reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_policy_trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--semantic_index_diagnostics", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--config_path", default=None)
    parser.add_argument(
        "--semantic_shared_prefix_repeats",
        type=int,
        default=4,
        help="Repeat a long common semantic serving scaffold this many times before each semantic prompt.",
    )
    parser.add_argument(
        "--semantic_shared_prefix_mode",
        choices=["common_scaffold", "variant_scaffold"],
        default="common_scaffold",
        help=(
            "common_scaffold reports only the common long scaffold as the exact shared prefix; "
            "variant_scaffold treats the long scaffold plus paraphrase variant as the shared prefix."
        ),
    )
    parser.add_argument("--early_layer_reuse_ratio", type=float, default=0.35)
    parser.add_argument("--logit_guard_threshold", type=float, default=0.08)
    parser.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    results_root = Path(args.results_root)
    logger = configure_logging(results_root)
    run_env = build_run_env(args)

    detected_config = {} if args.dry_run else apply_detected_config(log=True)
    jobs = list(iter_jobs(args))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "blackwell_semantic_n128_isolated",
        "models": args.models,
        "normal_models": NORMAL_MODELS,
        "large_models": LARGE_MODELS,
        "gemma4_models": GEMMA4_MODELS,
        "datasets": args.datasets,
        "prompt_modes": args.prompt_modes,
        "engines": args.engines,
        "seeds": args.seeds,
        "n_requests": args.n_requests,
        "semantic_reuse": "allow_unsafe_semantic_kv_reuse" if args.allow_unsafe_semantic_kv_reuse else "semantic_index_only",
        "semantic_shared_prefix_repeats": args.semantic_shared_prefix_repeats,
        "semantic_shared_prefix_mode": args.semantic_shared_prefix_mode,
        "isolation": "one_python_subprocess_per_model_dataset_prompt_mode_engine_cell",
        "total_jobs": len(jobs),
        "auto_detected_config": detected_config,
    }
    (results_root / "_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("=" * 80)
    logger.info(
        "BLACKWELL SEMANTIC N128 ISOLATED START models=%d datasets=%d modes=%d engines=%d seeds=%s jobs=%d",
        len(args.models),
        len(args.datasets),
        len(args.prompt_modes),
        len(args.engines),
        args.seeds,
        len(jobs),
    )
    logger.info("=" * 80)

    failed: list[dict[str, Any]] = []
    job_results: list[dict[str, Any]] = []
    started = time.time()
    for index, job in enumerate(jobs, start=1):
        output_dir = output_dir_for(results_root, job["model"], job["prompt_mode"], job["seed"], job["dataset"], job["engine"])
        log_prefix = log_prefix_for(results_root, job["model"], job["prompt_mode"], job["seed"], job["dataset"], job["engine"])
        logger.info(
            "JOB %d/%d model=%s dataset=%s mode=%s engine=%s seed=%s",
            index,
            len(jobs),
            job["model"],
            job["dataset"],
            job["prompt_mode"],
            job["engine"],
            job["seed"],
        )
        if args.skip_completed and is_completed(output_dir, job["engine"]):
            logger.info("     SKIP completed")
            result = {**job, "returncode": "skipped_completed", "elapsed_sec": 0.0, "output_dir": str(output_dir)}
            job_results.append(result)
            continue

        cmd = build_cmd(output_dir, job, args)
        returncode, elapsed = run_cmd(cmd, log_prefix, logger, args.dry_run, run_env)
        result = {**job, "returncode": returncode, "elapsed_sec": round(elapsed, 3), "output_dir": str(output_dir)}
        job_results.append(result)
        (results_root / "_job_results.json").write_text(json.dumps(job_results, indent=2), encoding="utf-8")

        if returncode != 0 and returncode != "dry_run":
            failed.append(result)
            if not args.continue_on_failure:
                break

    finalize_results(results_root)
    (results_root / "_job_results.json").write_text(json.dumps(job_results, indent=2), encoding="utf-8")
    logger.info("=" * 80)
    logger.info("DONE elapsed=%.1fs failed=%d/%d", time.time() - started, len(failed), len(jobs))
    logger.info("Saved %s", results_root / "all_results.csv")
    logger.info("Saved %s", results_root / "comparisons_vs_no_cache.csv")
    logger.info("Saved %s", results_root / "reuse_path_breakdown.csv")
    logger.info("=" * 80)
    return 1 if failed and not args.continue_on_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
