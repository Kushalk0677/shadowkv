# Blackwell Semantic-Reuse n=128 Pipeline

This pipeline is for running the Hugging Face backend on an RTX PRO 6000
Blackwell machine with semantic KV reuse enabled. It is designed as a clean
handoff package: one seed by default, the core ShadowKV engines, and one Python
subprocess per benchmark cell.

## Quick Start

Unzip or copy this package into the `v10` repo root on the Blackwell machine.
After unpacking, the repo should contain:

```text
v10/blackwell_semantic_n128/run_blackwell_semantic_n128.py
v10/blackwell_semantic_n128/README.md
```

Run these commands from the `v10` repo root.

First verify that the script expands into the expected commands:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py --dry_run
```

Then run a small real smoke test:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --models gpt2 \
  --datasets ag_news \
  --n_requests 16 \
  --no-measure_energy \
  --results_root results_blackwell_semantic_smoke
```

If the smoke test finishes, run the full default sweep:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py
```

The full default run is:

```text
11 models x 10 datasets x 1 prompt mode x 1 seed x 3 engines = 330 isolated subprocess jobs
```

Default output goes to:

```text
results_blackwell_semantic_n128/
```

## What Was Added

New self-contained Blackwell subfolder:

```text
blackwell_semantic_n128/
blackwell_semantic_n128/run_blackwell_semantic_n128.py
blackwell_semantic_n128/README.md
```

Default output directory:

```text
results_blackwell_semantic_n128/
```

The runner writes:

```text
results_blackwell_semantic_n128/_run_manifest.json
results_blackwell_semantic_n128/_sweep.log
results_blackwell_semantic_n128/_job_results.json
results_blackwell_semantic_n128/_logs/
results_blackwell_semantic_n128/all_results.csv
results_blackwell_semantic_n128/comparisons_vs_no_cache.csv
results_blackwell_semantic_n128/reuse_path_breakdown.csv
results_blackwell_semantic_n128/**/*.jsonl
results_blackwell_semantic_n128/<model>/semantic/seed_<seed>/<dataset>/<engine>/benchmark_*.json
```

## What It Runs

The full run uses the Hugging Face backend only:

```text
backend = hf
device = cuda
dtype = float16
```

Default request count:

```text
n_requests = 128
```

Default prompt mode:

```text
semantic
```

Default engines:

```text
no_cache
shadow_kv
shadow_kv_plus
```

These are the core HF engines for the paper comparison. Backend-specific
runtime adapters such as vLLM, SGLang, and LMCache are intentionally not part of
this handoff script.

Semantic reuse is enabled with:

```text
allow_unsafe_semantic_kv_reuse = true
```

The default run also uses longer shared semantic scaffolds:

```text
semantic_shared_prefix_repeats = 4
semantic_shared_prefix_mode = common_scaffold
```

This makes the benchmark exercise longer reusable prefixes, where HF KV reuse is
more likely to beat `past_key_values` overhead. The setting is explicit in
`_run_manifest.json` and can be disabled with:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --semantic_shared_prefix_repeats 0
```

Read exact scaffold reuse and semantic partial reuse separately. Use
`reuse_path_breakdown.csv`: `exact_scaffold_hits` means repeated exact scaffold
reuse executed, while `semantic_partial_hits` means approximate semantic partial
KV reuse actually executed.

## Default Model Matrix

The runner covers the five normal paper models:

```text
gpt2
TinyLlama/TinyLlama-1.1B-Chat-v1.0
Qwen/Qwen2.5-1.5B-Instruct
google/gemma-2b-it
microsoft/Phi-3-mini-4k-instruct
```

It also covers larger Qwen2.5 models up to 32B:

```text
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-32B-Instruct
```

It also includes only the two requested Gemma 4 models:

```text
google/gemma-4-12B-it
google/gemma-4-26b-a4b-it
```

The Gemma 4 12B target uses the official Hugging Face slug
`google/gemma-4-12B-it`.

Default datasets:

```text
ag_news
alpaca_eval
banking77
cnn_dailymail
daily_dialog
dolly
oasst1
samsum
ultrachat
xsum
```

Default seeds:

```text
42
```

Only one seed is used by default for this handoff. Add more seeds only after the
first full run completes cleanly.

## Isolation Model

The runner launches one subprocess for each:

```text
model x dataset x prompt_mode x seed x engine
```

That means `no_cache`, `shadow_kv`, and `shadow_kv_plus` do not share a live
backend, warmed kernels, allocator state, or previous KV tensors. This is slower
than keeping one model loaded, but it is the cleaner version to send to someone
else because every engine cell is isolated and restartable.

Each subprocess writes its own stdout/stderr files under:

```text
results_blackwell_semantic_n128/_logs/
```

The run is resumable. Completed engine cells are skipped by default.

## Why This Should Avoid The P100 OOM

The older P100 run failed on a 12 GB P100 because larger HF models could exhaust
VRAM or leave the allocator fragmented. This runner is intended for the RTX PRO
6000 Blackwell machine, which has much more memory, and it isolates each engine
cell in its own process. If a cell fails, the failure is contained to that cell
and the next process starts cleanly.

## Relationship To The Transferred RTX 6000 Bundle

The transferred Blackwell folder already contains `experiments/runall_for_6000.py`
and `RUNALL_FOR_6000_GUIDE.md`. That runner is a Qwen 14B isolation sweep:
one engine per subprocess, 256 requests, all prompt modes, and output under
`results_6000_qwen14b/`.

This subfolder is separate on purpose. It is the n=128 semantic-reuse sweep
requested for the paper discussion: multi-model, semantic mode by default, one
seed by default, all three core HF engines, isolated engine subprocesses, and
output under `results_blackwell_semantic_n128/`.

## Setup On The Blackwell Machine

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Recommended environment:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
```

If the machine should stay offline:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Check the GPU:

```bash
nvidia-smi
```

## Dry Run

Before spending GPU time:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py --dry_run
```

This prints the isolated benchmark commands without running the benchmarks.
It should show commands for `no_cache`, `shadow_kv`, and `shadow_kv_plus`.

## Smoke Run

Run one small model, one dataset:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --models gpt2 \
  --datasets ag_news \
  --seeds 42 \
  --n_requests 16 \
  --no-measure_energy \
  --results_root results_blackwell_semantic_smoke
```

Success means a benchmark JSON appears under:

```text
results_blackwell_semantic_smoke/gpt2/semantic/seed_42/ag_news/no_cache/
results_blackwell_semantic_smoke/gpt2/semantic/seed_42/ag_news/shadow_kv/
results_blackwell_semantic_smoke/gpt2/semantic/seed_42/ag_news/shadow_kv_plus/
```

## Main Run

Run the full default n=128 semantic matrix:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py
```

Recommended tmux usage:

```bash
tmux new -s blackwell_semantic
python blackwell_semantic_n128/run_blackwell_semantic_n128.py
```

Detach:

```text
Ctrl-b d
```

Reattach:

```bash
tmux attach -t blackwell_semantic
```

The runner is resumable. If interrupted, run the same command again. Completed
cells are skipped by default.

## Exact Command To Send

If sending only one instruction to the Blackwell operator, send this:

```bash
cd /path/to/v10
python blackwell_semantic_n128/run_blackwell_semantic_n128.py --dry_run
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --models gpt2 \
  --datasets ag_news \
  --n_requests 16 \
  --no-measure_energy \
  --results_root results_blackwell_semantic_smoke
python blackwell_semantic_n128/run_blackwell_semantic_n128.py
```

If they use `tmux`, run the full command inside the session:

```bash
tmux new -s blackwell_semantic
python blackwell_semantic_n128/run_blackwell_semantic_n128.py
```

## Targeted Phi-3 Run

To specifically verify the Phi-3 case that failed on P100:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --models microsoft/Phi-3-mini-4k-instruct \
  --datasets ag_news alpaca_eval banking77 cnn_dailymail daily_dialog dolly oasst1 samsum ultrachat xsum \
  --seeds 42 \
  --results_root results_blackwell_phi3_semantic_n128
```

## Targeted Large-Model Run

To run only Qwen larger models:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --models Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct \
  --results_root results_blackwell_qwen_large_semantic_n128
```

## Targeted Gemma 4 Run

To run only the requested Gemma 4 models:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --models google/gemma-4-12B-it google/gemma-4-26b-a4b-it \
  --results_root results_blackwell_gemma4_semantic_n128
```

If Hugging Face access fails, check network access and Hub authentication before
starting the benchmark.

## Optional Additional Modes

The default is semantic-only. If you also want raw and templated comparison:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --prompt_modes raw templated semantic \
  --results_root results_blackwell_all_modes_n128
```

This takes about three times as many cells.

## Optional Literature Baselines

Do not run literature/runtime baselines through this HF handoff script. This
script is only for the in-repo Hugging Face ShadowKV comparison.

The main harness has a larger engine registry, but this Blackwell handoff uses
only:

```text
no_cache
shadow_kv
shadow_kv_plus
```

The literature-accurate runtime baselines live separately under:

```text
literature_accurate_baselines/
```

The most relevant optional baselines are:

```text
vllm_apc
vllm_apc_shadowkv_plus
sglang_radix_attention
sglang_radix_attention_shadowkv_plus
lmcache
lmcache_shadowkv_plus
```

Those are run through:

```bash
python literature_accurate_baselines/run_runtime_cache_baseline.py --baseline <baseline_name>
```

There are also separate adapters/docs for:

```text
SGLang HiCache
KVFlow
oracle_future_reuse
```

Use them carefully. SGLang HiCache and LMCache should be measured only when the
actual runtime system is installed and launched. KVFlow needs a real
KVFlow-compatible serving endpoint or workflow trace setup. `oracle_future_reuse`
is an offline upper-bound oracle, not a competing deployed system.

## Important Columns To Inspect

Use:

```text
comparisons_vs_no_cache.csv
```

Primary columns:

```text
speedup_vs_no_cache_mean
speedup_vs_no_cache_p95
hit_rate
wasted_compute_ratio
policy_exact_total
fast_exact_path_hits
semantic_partial_hits
semantic_partial_reused_tokens_total
semantic_opportunity_plans_total
semantic_blocked_by_backend_total
semantic_quality_divergence_sum
semantic_quality_divergence_events
energy_reduction_vs_no_cache_pct
idle_adjusted_energy_reduction_vs_no_cache_pct
```

Interpretation rule:

Do not claim a win from semantic matches alone. First decide which reuse path
actually executed:

```text
fast_exact_path_hits > 0       # exact scaffold reuse
semantic_partial_hits > 0      # approximate semantic partial reuse
```

These are different claims. Exact scaffold hits are useful, but they are not
semantic partial KV reuse. After identifying the path, check whether it helped
or hurt latency:

```text
speedup_vs_no_cache_mean > 1.0
```

If energy columns are blank or null, do not make energy claims from that run.

## Expected Risks

### Hugging Face Backend Is Not A Production Runtime

The HF backend is useful for controlled policy experiments, but it is not the
same as vLLM, SGLang, or LMCache. Treat the results as in-process backend
evidence, not production-serving throughput evidence.

### Semantic Reuse Is Intentionally Unsafe In This Run

The runner enables approximate semantic KV reuse to measure what happens when it
actually executes. This is useful for evidence, but the paper should still frame
semantic execution as model-dependent and correctness-bounded.

### High Hit Rate Can Still Lose

A high hit rate does not guarantee speedup. Always use
`comparisons_vs_no_cache.csv` and compare latency/energy against the matched
`no_cache` cell. Use `reuse_path_breakdown.csv` to tell whether a hit-rate gain
came from exact scaffold reuse or semantic partial reuse.

### Qwen Models May Be Sensitive

Prior fidelity checks showed Qwen-style models can be sensitive to approximate
KV reuse in float16. If Qwen rows show semantic hits but worse latency or
quality divergence, that supports the need for safeguards.

### 32B May Be Slow

RTX PRO 6000 Blackwell has enough VRAM for the intended large-model cells, but
the 32B jobs can still take a long time. Run the smoke test first.

## What To Send Back

After the run, preserve:

```text
results_blackwell_semantic_n128/_run_manifest.json
results_blackwell_semantic_n128/_sweep.log
results_blackwell_semantic_n128/_job_results.json
results_blackwell_semantic_n128/_logs/**/*.stdout.txt
results_blackwell_semantic_n128/_logs/**/*.stderr.txt
results_blackwell_semantic_n128/all_results.csv
results_blackwell_semantic_n128/comparisons_vs_no_cache.csv
results_blackwell_semantic_n128/reuse_path_breakdown.csv
results_blackwell_semantic_n128/**/*.json
results_blackwell_semantic_n128/**/*.jsonl
```

To archive:

```bash
tar czf results_blackwell_semantic_n128.tgz results_blackwell_semantic_n128
```

Send back either the `.tgz` archive above or the whole
`results_blackwell_semantic_n128/` directory. The most important files are
`comparisons_vs_no_cache.csv`, `reuse_path_breakdown.csv`, `all_results.csv`,
`_run_manifest.json`, `_job_results.json`, and the per-cell logs under `_logs/`.
