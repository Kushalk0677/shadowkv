from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd


DEVICE = "cuda"
DTYPE = "float16"
MODELS = [
    "gpt2",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/gemma-2b-it",
]

RESULTS_ROOT = Path("results_n512")
LOGS_ROOT = RESULTS_ROOT / "_logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

N_REQUESTS = 16
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
PROMPT_MODES = ["raw", "templated", "semantic"]
MEAN_INTER_ARRIVAL_MS = "50"
MAX_ARRIVAL_SLEEP_MS = "500"

SKIP_COMPLETED = True
CONTINUE_ON_FAILURE = True
INCLUDE_SEMANTIC_ABLATIONS = True

# Your original HF sweep covers the in-repo baselines:
# no_cache, reactive_prefix_cache, greedy_prefix_cache,
# strict_reactive_prefix_cache, frequency_speculative, shadow_kv,
# shadow_kv_plus, shadow_kv_plus_best_latency, shadow_kv_plus_raw_observer,
# and semantic ablations when enabled.
RUN_MAIN_HF_BASELINES = True

# Literature-accurate runtime baselines. These require real vLLM/SGLang/LMCache
# installs and GPU capacity. They are run through literature_accurate_baselines.
RUN_RUNTIME_BASELINES = True
RUNTIME_BASELINES = [
    "vllm_apc",
    "vllm_apc_shadowkv_plus",
    "sglang_radix_attention",
    "sglang_radix_attention_shadowkv_plus",
    "lmcache",
    "lmcache_shadowkv_plus",
]

# If True, each runtime job launches and stops its server. This is simple and
# reproducible but slow. If False, the script attaches to a server you already
# started at the configured ports.
LAUNCH_RUNTIME_SERVER_PER_JOB = True
RUNTIME_HOST = "127.0.0.1"
RUNTIME_PORTS = {
    "vllm_apc": "8000",
    "vllm_apc_shadowkv_plus": "8000",
    "sglang_radix_attention": "30000",
    "sglang_radix_attention_shadowkv_plus": "30000",
    "lmcache": "8000",
    "lmcache_shadowkv_plus": "8000",
}
LMCACHE_ENGINE = "vllm"  # "vllm" or "sglang"


LOG_FILE = RESULTS_ROOT / "_sweep.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("sweep")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def is_completed(output_dir: Path, expected_engine: str | None = None) -> bool:
    if not output_dir.exists():
        return False
    json_files = list(output_dir.glob("*.json"))
    if not json_files:
        return False
    try:
        for path in sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True):
            data = json.loads(path.read_text(encoding="utf-8"))
            if "config" not in data:
                continue
            if expected_engine is None:
                return "no_cache" in data
            if expected_engine in data:
                return True
        return False
    except Exception:
        return False


def fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def run_cmd(cmd: list[str], log_prefix: Path) -> tuple[int, float]:
    log.info("CMD  " + " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.time() - start
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(log_prefix) + ".stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    Path(str(log_prefix) + ".stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    log.info(f"     {'OK' if proc.returncode == 0 else 'FAILED rc=' + str(proc.returncode)}  elapsed={fmt(elapsed)}")
    if proc.returncode != 0:
        log.error("ERROR:\n" + (proc.stderr or proc.stdout or "")[-2000:])
    return proc.returncode, elapsed


def main_hf_cmd(model: str, dataset: str, prompt_mode: str, seed: int, output_dir: Path) -> list[str]:
    cmd = [
        "python",
        "experiments/run_benchmark.py",
        "--backend",
        "hf",
        "--model",
        model,
        "--device",
        DEVICE,
        "--dtype",
        DTYPE,
        "--workload",
        "public_dataset",
        "--prompt_mode",
        prompt_mode,
        "--dataset",
        dataset,
        "--n_requests",
        str(N_REQUESTS),
        "--mean_inter_arrival_ms",
        MEAN_INTER_ARRIVAL_MS,
        "--max_arrival_sleep_ms",
        MAX_ARRIVAL_SLEEP_MS,
        "--seed",
        str(seed),
        "--include_experimental",
        "--output_dir",
        str(output_dir),
    ]
    if INCLUDE_SEMANTIC_ABLATIONS and prompt_mode == "semantic":
        cmd.append("--include_semantic_ablations")
    return cmd


def runtime_cmd(
    baseline: str,
    model: str,
    dataset: str,
    prompt_mode: str,
    seed: int,
    output_dir: Path,
) -> list[str]:
    cmd = [
        "python",
        "literature_accurate_baselines/run_runtime_cache_baseline.py",
        "--baseline",
        baseline,
        "--model",
        model,
        "--host",
        RUNTIME_HOST,
        "--port",
        RUNTIME_PORTS[baseline],
        "--workload",
        "public_dataset",
        "--prompt_mode",
        prompt_mode,
        "--dataset",
        dataset,
        "--n_requests",
        str(N_REQUESTS),
        "--mean_inter_arrival_ms",
        MEAN_INTER_ARRIVAL_MS,
        "--max_arrival_sleep_ms",
        MAX_ARRIVAL_SLEEP_MS,
        "--seed",
        str(seed),
        "--output_dir",
        str(output_dir),
        "--dtype",
        DTYPE,
    ]
    if baseline.startswith("lmcache"):
        cmd.extend(["--lmcache_engine", LMCACHE_ENGINE])
    if LAUNCH_RUNTIME_SERVER_PER_JOB:
        cmd.append("--launch_server")
    return cmd


def iter_jobs():
    for model in MODELS:
        tag = model_tag(model)
        for seed in SEEDS:
            for prompt_mode in PROMPT_MODES:
                for dataset in DATASETS:
                    if RUN_MAIN_HF_BASELINES:
                        yield {
                            "kind": "main_hf",
                            "expected_engine": None,
                            "model": model,
                            "tag": tag,
                            "seed": seed,
                            "prompt_mode": prompt_mode,
                            "dataset": dataset,
                        }
                    if RUN_RUNTIME_BASELINES:
                        for baseline in RUNTIME_BASELINES:
                            yield {
                                "kind": "runtime",
                                "baseline": baseline,
                                "expected_engine": baseline,
                                "model": model,
                                "tag": tag,
                                "seed": seed,
                                "prompt_mode": prompt_mode,
                                "dataset": dataset,
                            }


def output_paths(job: dict) -> tuple[Path, Path]:
    base = RESULTS_ROOT / job["tag"] / job["prompt_mode"] / f"seed_{job['seed']}" / job["dataset"]
    logs = LOGS_ROOT / job["tag"] / job["prompt_mode"] / f"seed_{job['seed']}" / job["dataset"]
    if job["kind"] == "main_hf":
        return base / "main_hf", logs / "main_hf"
    baseline = job["baseline"]
    return base / "runtime" / baseline, logs / "runtime" / baseline


def import_results() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_ROOT.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = data.get("config", {})
        caps = data.get("capabilities", {})
        runtime = data.get("runtime", {})
        for engine, metrics in data.items():
            if engine in ("config", "capabilities", "runtime") or not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "model": config.get("model"),
                    "dataset": config.get("dataset"),
                    "prompt_mode": config.get("resolved_prompt_mode") or config.get("prompt_mode"),
                    "seed": config.get("seed"),
                    "engine": engine,
                    "runtime_kind": runtime.get("runtime_kind"),
                    "literature_accurate_external_runtime": runtime.get("literature_accurate_external_runtime"),
                    "mean_latency_ms": metrics.get("mean_latency_ms"),
                    "p50_latency_ms": metrics.get("p50_latency_ms"),
                    "p95_latency_ms": metrics.get("p95_latency_ms"),
                    "speedup_vs_no_cache_mean": metrics.get("speedup_vs_no_cache_mean"),
                    "speedup_vs_no_cache_p95": metrics.get("speedup_vs_no_cache_p95"),
                    "hit_rate": metrics.get("hit_rate"),
                    "cached_tokens_total": metrics.get("cached_tokens_total"),
                    "cached_tokens_mean": metrics.get("cached_tokens_mean"),
                    "wasted_compute_ratio": metrics.get("wasted_compute_ratio"),
                    "reuse_attempts": metrics.get("reuse_attempts"),
                    "reuse_successes": metrics.get("reuse_successes"),
                    "speculative_hits": metrics.get("speculative_hits"),
                    "speculative_hit_rate": metrics.get("speculative_hit_rate"),
                    "wasted_compute_ms": metrics.get("wasted_compute_ms"),
                    "utility_proxy_ms": metrics.get("utility_proxy_ms"),
                    "policy_plans_total": metrics.get("policy_plans_total"),
                    "policy_bypass_total": metrics.get("policy_bypass_total"),
                    "policy_exact_total": metrics.get("policy_exact_total"),
                    "policy_semantic_partial_total": metrics.get("policy_semantic_partial_total"),
                    "admission_controller_enabled": metrics.get("admission_controller_enabled"),
                    "admission_plans_total": metrics.get("admission_plans_total"),
                    "admission_allow_total": metrics.get("admission_allow_total"),
                    "admission_bypass_total": metrics.get("admission_bypass_total"),
                    "admission_runtime_cache_resets": metrics.get("admission_runtime_cache_resets"),
                    "supports_external_kv": caps.get("supports_external_kv"),
                    "supports_native_prefix_caching": caps.get("supports_native_prefix_caching"),
                    "source_json": str(path),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    jobs = list(iter_jobs())
    total_jobs = len(jobs)
    sweep_start = time.time()
    failed_jobs = []
    results = {}

    log.info("=" * 80)
    log.info(
        f"SWEEP START  n_requests={N_REQUESTS}  models={len(MODELS)}  "
        f"datasets={len(DATASETS)}  modes={len(PROMPT_MODES)}  total_jobs={total_jobs}"
    )
    log.info(f"main_hf={RUN_MAIN_HF_BASELINES} runtime={RUN_RUNTIME_BASELINES} runtime_baselines={RUNTIME_BASELINES}")
    log.info("=" * 80)

    for job_id, job in enumerate(jobs, start=1):
        output_dir, log_prefix = output_paths(job)
        elapsed_total = time.time() - sweep_start
        remaining = (total_jobs - job_id) / max(job_id / max(elapsed_total, 1), 1e-9)
        baseline_label = job.get("baseline", "main_hf")

        log.info("")
        log.info("-" * 80)
        log.info(f"JOB {job_id}/{total_jobs}  ETA~{fmt(remaining)}")
        log.info(
            f"  kind={job['kind']} baseline={baseline_label} model={job['model']} "
            f"dataset={job['dataset']} mode={job['prompt_mode']} seed={job['seed']}"
        )
        log.info("-" * 80)

        if SKIP_COMPLETED and is_completed(output_dir, expected_engine=job["expected_engine"]):
            log.info("  -> skipping (already completed)")
            results[str(output_dir)] = {"returncode": "skipped_completed"}
            continue

        if job["kind"] == "main_hf":
            cmd = main_hf_cmd(job["model"], job["dataset"], job["prompt_mode"], job["seed"], output_dir)
        else:
            cmd = runtime_cmd(
                job["baseline"],
                job["model"],
                job["dataset"],
                job["prompt_mode"],
                job["seed"],
                output_dir,
            )

        rc, elapsed = run_cmd(cmd, log_prefix)
        results[str(output_dir)] = {"returncode": rc, "elapsed_sec": round(elapsed, 1)}
        if rc != 0:
            failed_jobs.append(
                f"kind={job['kind']} baseline={baseline_label} model={job['model']} "
                f"dataset={job['dataset']} mode={job['prompt_mode']} seed={job['seed']}"
            )
            if not CONTINUE_ON_FAILURE:
                log.error("Aborting.")
                return 1

    total_wall = time.time() - sweep_start
    log.info("")
    log.info("=" * 80)
    log.info(f"SWEEP DONE  wall={fmt(total_wall)}  failed={len(failed_jobs)}/{total_jobs}")
    for failed in failed_jobs:
        log.warning(f"  FAILED: {failed}")
    log.info("=" * 80)

    manifest = {
        "models": MODELS,
        "device": DEVICE,
        "dtype": DTYPE,
        "n_requests": N_REQUESTS,
        "prompt_modes": PROMPT_MODES,
        "datasets": DATASETS,
        "seeds": SEEDS,
        "run_main_hf_baselines": RUN_MAIN_HF_BASELINES,
        "run_runtime_baselines": RUN_RUNTIME_BASELINES,
        "runtime_baselines": RUNTIME_BASELINES,
        "launch_runtime_server_per_job": LAUNCH_RUNTIME_SERVER_PER_JOB,
        "total_jobs": total_jobs,
        "total_wall_seconds": round(total_wall, 1),
        "failed_jobs": failed_jobs,
    }
    (RESULTS_ROOT / "_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (RESULTS_ROOT / "_job_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    df = import_results()
    log.info(f"Imported {len(df)} result rows")
    if not df.empty:
        df.to_csv(RESULTS_ROOT / "n512_all_results.csv", index=False)
        log.info(f"Saved: {RESULTS_ROOT}/n512_all_results.csv")

    zip_path = shutil.make_archive("results_n512", "zip", RESULTS_ROOT)
    log.info(f"Zip ready: {zip_path}")
    log.info("scp user@host:" + zip_path + " .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
