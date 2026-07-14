# Phase 2 + Phase 3 Implementation Report

## Phase 2: Energy Instrumentation

Added `src/proactive_kv_cache/energy.py` with an NVML-first `NvidiaEnergyMeter`.

Implemented:

- `EnergySnapshot`
- `NvidiaEnergyMeter.snapshot()`
- `NvidiaEnergyMeter.delta(before, after)`
- `measure_idle_baseline()`
- Optional `nvidia-smi` fallback for instantaneous power/memory when NVML is unavailable

New benchmark CLI flags:

```bash
--measure_energy
--gpu_index 0
--idle_baseline_seconds 10
--energy_output_csv results/energy_summary.csv
```

Energy columns now emitted into the JSON summary and CSV summary:

- `energy_source`
- `energy_elapsed_s`
- `gpu_energy_j`
- `avg_power_w_from_energy`
- `idle_baseline_power_w`
- `idle_adjusted_gpu_energy_j`
- `gpu_joules_per_request`
- `idle_adjusted_joules_per_request`
- `gpu_joules_per_total_token`
- `energy_reduction_vs_no_cache_pct`
- `idle_adjusted_energy_reduction_vs_no_cache_pct`

Notes:

- NVML total-energy deltas are used when available.
- If total-energy counters are not exposed, the benchmark reports energy as unavailable instead of inventing values.
- Idle baseline subtraction is optional and should be used for paper-grade runs.

## Phase 3: Utility-Aware Admission

Added `src/proactive_kv_cache/utility_admission.py`.

Implemented:

- prefix-length buckets
- online full-prefill cost estimator
- online reuse-overhead estimator
- net-utility decisioning
- negative-utility bypass counters

Bucket ranges:

```text
0-64
65-128
129-256
257-512
513-1024
1025-2048
2049-4096
4097+
```

New ShadowKV++ Lite behavior:

```text
expected_saved_ms = estimated_full_prefill_ms_per_token * prefix_tokens
expected_cost_ms  = estimated_reuse_overhead_ms + suffix_penalty
net_utility_ms    = expected_saved_ms - expected_cost_ms

admit reuse only if net_utility_ms >= utility_min_net_saved_ms
```

New CLI flags:

```bash
--disable_utility_admission
--utility_min_net_saved_ms 0.0
```

New telemetry/counters:

- `utility_admission_enabled`
- `utility_admission_checks_total`
- `utility_admission_admit_total`
- `negative_utility_bypass_total`
- `utility_expected_saved_ms_total`
- `utility_expected_cost_ms_total`
- `utility_net_ms_total`
- `utility_bucket_*_observations`
- `utility_bucket_*_full_ms_per_token`
- `utility_bucket_*_reuse_overhead_ms`
- `utility_bucket_*_admitted`
- `utility_bucket_*_negative_bypass`

## CSV Output

`run_benchmark.py` now writes a per-engine CSV when `--measure_energy` or `--energy_output_csv` is used.

The CSV includes latency, p95, speedup, energy, utility admission, bypass, and hot-path timing columns.

## Validation

Full test suite:

```text
76 passed, 1 skipped
```

The skipped test is the existing optional slow Hugging Face KV correctness test.

Smoke run:

```bash
python experiments/run_benchmark.py \
  --backend fake \
  --engines no_cache shadow_kv_plus_lite \
  --workload synthetic \
  --variant high_skew \
  --n_requests 16 \
  --disable_arrival_simulation \
  --min_reuse_prefix_tokens 3 \
  --utility_min_net_saved_ms 0.0 \
  --measure_energy \
  --output_dir /mnt/data/phase23_smoke
```

Observed smoke signal:

```text
shadow_kv_plus_lite speedup vs no_cache mean: 1.52x
utility_admission_checks_total: 13
utility_admission_admit_total: 13
negative_utility_bypass_total: 0
lite_fast_path_total: 13
```

Negative-utility bypass smoke run:

```bash
python experiments/run_benchmark.py \
  --backend fake \
  --engines shadow_kv_plus_lite \
  --workload synthetic \
  --variant high_skew \
  --n_requests 8 \
  --disable_arrival_simulation \
  --min_reuse_prefix_tokens 3 \
  --utility_min_net_saved_ms 9999
```

Observed:

```text
negative_utility_bypass_total: 5
utility_admission_checks_total: 5
lite_fast_path_total: 0
```

## Recommended paper-grade benchmark command

Use this style for CUDA/HF runs:

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model qwen25_15b \
  --device cuda \
  --dtype float16 \
  --workload public_dataset \
  --dataset samsum \
  --prompt_mode rag \
  --n_requests 512 \
  --engines no_cache shadow_kv_plus_lite shadow_kv_plus reactive_prefix_cache greedy_prefix_cache \
  --disable_arrival_simulation \
  --min_reuse_prefix_tokens 96 \
  --utility_min_net_saved_ms 0.0 \
  --measure_energy \
  --idle_baseline_seconds 10 \
  --output_dir results_phase23
```

For threshold sweeps, repeat with:

```text
--min_reuse_prefix_tokens 64
--min_reuse_prefix_tokens 96
--min_reuse_prefix_tokens 128
--min_reuse_prefix_tokens 192
--min_reuse_prefix_tokens 256
```
