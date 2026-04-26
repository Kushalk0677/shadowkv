from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_DATASETS = [
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


def safe_tag(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace(".", "_")


def run(cmd: list[str], log_prefix: Path) -> int:
    print("\nRunning:")
    print(" ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.time() - start
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    log_prefix.with_suffix(".stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    print(f"returncode={proc.returncode} elapsed_sec={elapsed:.1f}")
    if proc.returncode != 0:
        print((proc.stderr or proc.stdout or "")[-2500:])
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--backend", choices=["fake", "hf"], default="hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--n_requests", type=int, default=128)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--early_layer_ratios", nargs="*", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--logit_guard_thresholds", nargs="*", type=float, default=[0.04, 0.08, 0.12, 0.20])
    parser.add_argument("--output_root", default="results_semantic_correctness_ablations")
    parser.add_argument("--continue_on_failure", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    logs = root / "_logs"
    model_tag = safe_tag(args.model)

    manifest = {
        "model": args.model,
        "backend": args.backend,
        "device": args.device,
        "dtype": args.dtype,
        "datasets": args.datasets,
        "n_requests": args.n_requests,
        "seeds": args.seeds,
        "early_layer_ratios": args.early_layer_ratios,
        "logit_guard_thresholds": args.logit_guard_thresholds,
        "jobs": [],
    }

    job = 0
    total = len(args.datasets) * len(args.seeds) * (1 + len(args.early_layer_ratios) + len(args.logit_guard_thresholds))
    for seed in args.seeds:
        for dataset in args.datasets:
            # Baseline run: safe + all ablation engines at default settings.
            job += 1
            out = root / model_tag / "baseline_all_ablations" / f"seed_{seed}" / dataset
            log_prefix = logs / model_tag / "baseline_all_ablations" / f"seed_{seed}" / dataset
            print(f"\nJOB {job}/{total}: {dataset} seed={seed} baseline_all_ablations")
            cmd = [
                sys.executable, "experiments/run_benchmark.py",
                "--backend", args.backend,
                "--model", args.model,
                "--device", args.device,
                "--dtype", args.dtype,
                "--workload", "public_dataset",
                "--prompt_mode", "semantic",
                "--dataset", dataset,
                "--n_requests", str(args.n_requests),
                "--mean_inter_arrival_ms", "50",
                "--max_arrival_sleep_ms", "500",
                "--seed", str(seed),
                "--include_experimental",
                "--include_semantic_ablations",
                "--output_dir", str(out),
            ]
            rc = run(cmd, log_prefix)
            manifest["jobs"].append({"dataset": dataset, "seed": seed, "type": "baseline_all_ablations", "returncode": rc, "output_dir": str(out)})
            if rc != 0 and not args.continue_on_failure:
                raise SystemExit(rc)

            for ratio in args.early_layer_ratios:
                job += 1
                label = f"early_layer_{ratio:.2f}".replace(".", "p")
                out = root / model_tag / label / f"seed_{seed}" / dataset
                log_prefix = logs / model_tag / label / f"seed_{seed}" / dataset
                print(f"\nJOB {job}/{total}: {dataset} seed={seed} early_layer_ratio={ratio}")
                cmd2 = list(cmd)
                cmd2[cmd2.index("--output_dir") + 1] = str(out)
                cmd2.extend(["--early_layer_reuse_ratio", str(ratio)])
                rc = run(cmd2, log_prefix)
                manifest["jobs"].append({"dataset": dataset, "seed": seed, "type": label, "returncode": rc, "output_dir": str(out)})
                if rc != 0 and not args.continue_on_failure:
                    raise SystemExit(rc)

            for thr in args.logit_guard_thresholds:
                job += 1
                label = f"logit_guard_{thr:.2f}".replace(".", "p")
                out = root / model_tag / label / f"seed_{seed}" / dataset
                log_prefix = logs / model_tag / label / f"seed_{seed}" / dataset
                print(f"\nJOB {job}/{total}: {dataset} seed={seed} logit_guard_threshold={thr}")
                cmd3 = list(cmd)
                cmd3[cmd3.index("--output_dir") + 1] = str(out)
                cmd3.extend(["--logit_guard_threshold", str(thr)])
                rc = run(cmd3, log_prefix)
                manifest["jobs"].append({"dataset": dataset, "seed": seed, "type": label, "returncode": rc, "output_dir": str(out)})
                if rc != 0 and not args.continue_on_failure:
                    raise SystemExit(rc)

    root.mkdir(parents=True, exist_ok=True)
    (root / "_semantic_ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest to {root / '_semantic_ablation_manifest.json'}")


if __name__ == "__main__":
    main()
