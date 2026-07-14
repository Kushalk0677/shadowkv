from __future__ import annotations

"""Run the ShadowKV++ semantic/paraphrase novelty benchmark.

This script isolates the claim that ShadowKV++ can detect semantically-equivalent
serving scaffolds even when exact prefix strings differ. Exact prefix caches should
have limited benefit in this mode; ShadowKV++ records semantic opportunity metrics
and, on FakeBackend, can execute approximate partial reuse for controlled ablation.
"""

import subprocess
from pathlib import Path

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct"]
DATASETS = ["ag_news", "banking77", "alpaca_eval", "samsum", "cnn_dailymail"]
SEEDS = [42]
N_REQUESTS = 32
DEVICE = "cuda"
DTYPE = "float16"


def run(cmd: list[str]) -> int:
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100)
    return subprocess.run(cmd).returncode


def main() -> None:
    results: dict[str, dict[str, int]] = {}
    for model in MODELS:
        model_tag = model.replace("/", "_").replace(":", "_")
        results[model] = {}
        for seed in SEEDS:
            for dataset in DATASETS:
                out = Path("results_semantic_novelty") / model_tag / f"seed_{seed}" / dataset
                cmd = [
                    "python", "experiments/run_benchmark.py",
                    "--backend", "hf",
                    "--model", model,
                    "--device", DEVICE,
                    "--dtype", DTYPE,
                    "--workload", "public_dataset",
                    "--prompt_mode", "semantic",
                    "--dataset", dataset,
                    "--n_requests", str(N_REQUESTS),
                    "--mean_inter_arrival_ms", "50",
                    "--max_arrival_sleep_ms", "500",
                    "--seed", str(seed),
                    "--include_experimental",
                    "--output_dir", str(out),
                ]
                results[model][f"{dataset}/seed_{seed}"] = run(cmd)
    print("\nSummary:")
    for model, model_results in results.items():
        print(model, model_results)


if __name__ == "__main__":
    main()
