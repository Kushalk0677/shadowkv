# RTX 6000 Blackwell Run Guide

This guide is for another AI agent running the Qwen 14B ShadowKV++ sweep on an NVIDIA RTX PRO 6000 Blackwell workstation.

The only script to run is:

```bash
python experiments/runall_for_6000.py
```

Do not use `experiments/run_all_baselines_sweep.py` for this run. Do not call `experiments/run_benchmark.py` manually for this run. The dedicated script already applies the intended benchmark structure.

## What The Script Runs

`experiments/runall_for_6000.py` runs:

```text
Model: Qwen/Qwen2.5-14B-Instruct
Backend: Hugging Face
Device: CUDA
Dtype: float16
Requests per run: 256
Datasets: all 10 public datasets configured in v10
Prompt modes: raw, templated, semantic, rag
Engines: no_cache, shadow_kv_plus
Energy measurement: enabled
Idle baseline: 10 seconds
Isolation: one engine per Python subprocess
Automatic hardware config: enabled before jobs start
```

Total jobs:

```text
10 datasets * 4 prompt modes * 2 engines = 80 jobs
```

## Why This Script Is Required

Large-model benchmark results should isolate engines. Reusing one live model/backend across engines can contaminate measurements through warmed kernels, leftover cache state, backend cache state, allocator fragmentation, or previous KV tensors.

The new runner avoids that by launching each engine run in its own Python process. This is the intended execution path for the RTX 6000 Blackwell 14B sweep.

Before launching jobs, the runner also calls the same hardware detection hook used by the old sweep script:

```text
experiments.hw_detect.apply_detected_config(log=True)
```

This updates `config/config.yaml` with detected GPU memory and PCIe bandwidth so the benchmark does not use stale hardware defaults.

## Before Running

From the v10 repo root, confirm the GPU is visible:

```bash
nvidia-smi
```

Set the PyTorch allocator config:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If Hugging Face auth is needed, ensure `HF_TOKEN` is available in the environment.

Stop any other process using significant GPU memory before starting the sweep.

## Run Command

From the v10 repo root:

```bash
python experiments/runall_for_6000.py
```

Let it run to completion. If the process is interrupted, run the same command again. The script skips completed jobs automatically.

## Output Location

All outputs are written under:

```text
results_6000_qwen14b/
```

Primary files:

```text
results_6000_qwen14b/_sweep.log
results_6000_qwen14b/_run_manifest.json
results_6000_qwen14b/_job_results.json
results_6000_qwen14b/all_results.csv
results_6000_qwen14b/comparisons_vs_no_cache.csv
```

`_run_manifest.json` records the detected hardware config under:

```text
auto_detected_config
```

Per-job logs are under:

```text
results_6000_qwen14b/_logs/
```

Per-job benchmark JSONs are under:

```text
results_6000_qwen14b/Qwen_Qwen2_5-14B-Instruct/
```

## What To Review

Use this file for the main result comparison:

```text
results_6000_qwen14b/comparisons_vs_no_cache.csv
```

This file compares each `shadow_kv_plus` run against the matching `no_cache` run for the same dataset, prompt mode, seed, model, backend, device, dtype, and request count.

Important columns:

```text
speedup_vs_no_cache_mean
speedup_vs_no_cache_p95
hit_rate
reuse_successes
reused_prefix_tokens_total
energy_reduction_vs_no_cache_pct
idle_adjusted_energy_reduction_vs_no_cache_pct
```

Use this file for raw per-engine metrics:

```text
results_6000_qwen14b/all_results.csv
```

## What Was Wrong With The Earlier Data

The previous tester bundle was useful, but several conclusions in the written report did not match the JSON metrics. Use these checks when reviewing the new sweep.

### 1. Do Not Trust Hit Rate Alone

The earlier templated Qwen 14B run showed:

```text
hit_rate ~= 98.4%
reuse_successes = 63
```

but latency was worse than `no_cache`:

```text
no_cache mean ~= 33.07 ms
shadow_kv_plus mean ~= 39.66 ms
```

That means ShadowKV++ found reusable prefixes, but the HF backend reuse path cost more than the saved prefill. For the new sweep, always check speedup and energy, not only cache hits.

Look for:

```text
speedup_vs_no_cache_mean > 1.0
speedup_vs_no_cache_p95 > 1.0
```

### 2. Separate Engine Runs Need Matched Comparison

The earlier report mixed single-engine and multi-engine results. A single-engine `shadow_kv_plus` JSON does not automatically contain speedup versus `no_cache`.

The new script fixes this by writing:

```text
comparisons_vs_no_cache.csv
```

Use this file for speedup. It pairs each `shadow_kv_plus` run with the matching `no_cache` run using the same dataset, prompt mode, seed, model, backend, device, dtype, and request count.

### 3. The Old Report Misstated Backend Sharing

The previous report claimed that running multiple engines with `--engines no_cache shadow_kv shadow_kv_plus` shared one `HuggingFaceBackend`.

That was incorrect. `experiments/run_benchmark.py` creates a fresh backend inside the per-engine loop. This matters because large models can OOM or fragment memory if many engines run inside one benchmark invocation.

The new script avoids this ambiguity by launching one engine per process.

### 4. Semantic Matches Are Not The Same As Semantic KV Reuse

The earlier semantic run showed many semantic index matches:

```text
semantic_queries_total = 64
semantic_matches_total = 250
```

but:

```text
semantic_partial_hits = 0
policy_semantic_partial_total = 0
hit_rate = 0
```

So the semantic index was active, but safe semantic KV reuse did not actually happen. In the new sweep, if semantic rows show matches, still check whether there are actual semantic partial hits or latency gains.

Look for:

```text
policy_semantic_partial_total
semantic_partial_hits
speedup_vs_no_cache_mean
```

### 5. Energy Was Missing Before

The earlier tester bundle had:

```text
measure_energy = false
idle_baseline_seconds = 0.0
```

So it had no valid energy result.

The new script enables energy measurement. If the energy columns are blank, report that energy was unavailable. Do not infer energy savings from latency alone.

Look for:

```text
gpu_joules_per_request
idle_adjusted_joules_per_request
energy_reduction_vs_no_cache_pct
idle_adjusted_energy_reduction_vs_no_cache_pct
```

### 6. Raw Mode Usually Should Not Win

The earlier raw-mode results showed little or no reusable prefix overlap. That is expected. If raw mode does not improve, that is not automatically a failure. The important question is whether ShadowKV++ avoids harmful reuse and stays close to `no_cache`.

For raw rows, check:

```text
speedup_vs_no_cache_mean close to 1.0
hit_rate low
wasted_compute_ratio low
```

## Interpretation Rule

Do not claim ShadowKV++ wins from hit rate alone.

The result is a win only when the matched `shadow_kv_plus` row improves latency and/or energy versus `no_cache` in:

```text
comparisons_vs_no_cache.csv
```

For each row, check:

```text
speedup_vs_no_cache_mean > 1.0
```

and, when energy is available:

```text
energy_reduction_vs_no_cache_pct > 0
```

## Energy Notes

The script enables energy measurement. If energy fields are blank or null, the GPU/driver/NVML path did not expose usable energy counters for that run. In that case, report latency and reuse metrics only. Do not make energy claims from missing energy fields.

## Failure Checks

If a job fails, inspect:

```text
results_6000_qwen14b/_job_results.json
results_6000_qwen14b/_logs/
```

If CUDA OOM appears, check for competing GPU processes:

```bash
nvidia-smi
```

Then rerun the same command:

```bash
python experiments/runall_for_6000.py
```

The script will continue from completed jobs.

## Deliverables

After completion, preserve:

```text
results_6000_qwen14b/_sweep.log
results_6000_qwen14b/_run_manifest.json
results_6000_qwen14b/_job_results.json
results_6000_qwen14b/all_results.csv
results_6000_qwen14b/comparisons_vs_no_cache.csv
results_6000_qwen14b/**/*.json
results_6000_qwen14b/_logs/**/*.stdout.txt
results_6000_qwen14b/_logs/**/*.stderr.txt
```
