# ShadowKV Blackwell Semantic n=128 Handoff

This is a restricted, runnable ShadowKV repo bundle for the RTX PRO 6000
Blackwell semantic-reuse sweep.

## What This Repo Runs

The main Blackwell command runs:

```text
backend = hf
device = cuda
dtype = float16
n_requests = 128
prompt_mode = semantic
seed = 42
engines = no_cache, shadow_kv, shadow_kv_plus
semantic_shared_prefix_repeats = 4
semantic_shared_prefix_mode = common_scaffold
```

The default matrix is:

```text
11 models x 10 datasets x 1 prompt mode x 1 seed x 3 engines = 330 isolated subprocess jobs
```

Each engine cell runs in its own Python subprocess. This is intentional: it
keeps `no_cache`, `shadow_kv`, and `shadow_kv_plus` from sharing live model
state, warmed kernels, allocator state, or leftover KV tensors.

## Run Commands

Run all commands from the repo root, the directory containing `pyproject.toml`.

Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Recommended GPU environment:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
```

Check the commands without running GPU work:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py --dry_run
```

Run a small smoke test:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --models gpt2 \
  --datasets ag_news \
  --n_requests 16 \
  --no-measure_energy \
  --results_root results_blackwell_semantic_smoke
```

Run the full sweep:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py
```

Recommended `tmux` usage:

```bash
tmux new -s blackwell_semantic
python blackwell_semantic_n128/run_blackwell_semantic_n128.py
```

Detach with `Ctrl-b d`. Reattach with:

```bash
tmux attach -t blackwell_semantic
```

## Models

Normal paper models:

```text
gpt2
TinyLlama/TinyLlama-1.1B-Chat-v1.0
Qwen/Qwen2.5-1.5B-Instruct
google/gemma-2b-it
microsoft/Phi-3-mini-4k-instruct
```

Large models:

```text
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-32B-Instruct
```

Requested Gemma 4 models:

```text
google/gemma-4-12B-it
google/gemma-4-26b-a4b-it
```

The Gemma 4 12B target uses the official Hugging Face slug
`google/gemma-4-12B-it`.

## Results To Send Back

The default output directory is:

```text
results_blackwell_semantic_n128/
```

The most important files are:

```text
results_blackwell_semantic_n128/_run_manifest.json
results_blackwell_semantic_n128/_sweep.log
results_blackwell_semantic_n128/_job_results.json
results_blackwell_semantic_n128/all_results.csv
results_blackwell_semantic_n128/comparisons_vs_no_cache.csv
results_blackwell_semantic_n128/reuse_path_breakdown.csv
results_blackwell_semantic_n128/_logs/**/*.stdout.txt
results_blackwell_semantic_n128/_logs/**/*.stderr.txt
results_blackwell_semantic_n128/**/*.json
results_blackwell_semantic_n128/**/*.jsonl
```

Use `comparisons_vs_no_cache.csv` for matched latency and energy comparisons.
Use `reuse_path_breakdown.csv` to separate the two reuse mechanisms:

```text
exact_scaffold_hits
semantic_partial_policy_total
semantic_partial_hits
semantic_opportunity_plans_total
path_reading
```

`exact_scaffold_hits` means repeated exact scaffold KV reuse executed.
`semantic_partial_hits` means approximate semantic partial KV reuse actually
executed. Do not describe exact scaffold reuse as semantic partial reuse.

Archive command:

```bash
tar czf results_blackwell_semantic_n128.tgz results_blackwell_semantic_n128
```

## Literature Baselines

The Blackwell semantic runner is only the in-repo Hugging Face ShadowKV
comparison. Runtime-system baselines are separate.

The relevant optional baselines live in:

```text
literature_accurate_baselines/
```

Useful baseline names:

```text
vllm_apc
vllm_apc_shadowkv_plus
sglang_radix_attention
sglang_radix_attention_shadowkv_plus
lmcache
lmcache_shadowkv_plus
```

Those are launched through:

```bash
python literature_accurate_baselines/run_runtime_cache_baseline.py --baseline <baseline_name>
```

SGLang HiCache, KVFlow, and `oracle_future_reuse` are also documented there,
but they should not be mixed into the main HF semantic run. HiCache/LMCache need
their real runtimes installed. KVFlow needs a real KVFlow-compatible endpoint or
workflow trace setup. `oracle_future_reuse` is an offline upper bound, not a
deployed system baseline.

## What Was Excluded

This restricted repo bundle excludes generated results, caches, old fidelity
scratch directories, `.git`, notebook scratch artifacts, and bulky profiling
files. It includes the code and docs needed to run the Blackwell semantic n=128
pipeline and the optional literature-baseline adapters.
