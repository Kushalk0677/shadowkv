# Reproducing Blackwell Semantic n=128 Runs

Use `experiments/run_blackwell_semantic_n128.py` for the RTX PRO 6000 Blackwell semantic-reuse sweep. This is the cleaned public version of the local handoff package; the old zip packages are not required.

## Smoke Test

```bash
python experiments/run_blackwell_semantic_n128.py \
  --models gpt2 \
  --datasets ag_news \
  --n_requests 16 \
  --no-measure_energy \
  --results_root results_blackwell_semantic_smoke
```

## Full Default Run

```bash
python experiments/run_blackwell_semantic_n128.py
```

Default matrix:

```text
11 models x 10 datasets x 1 prompt mode x 1 seed x 3 engines = 330 subprocess jobs
```

Default engines:

```text
no_cache
shadow_kv
shadow_kv_plus
```

Default prompt mode is `semantic`, and approximate semantic KV reuse is enabled by default so the run can measure executed semantic reuse. Treat those results as experimental evidence, not a production-safety claim.

## Large Model Subsets

Only Qwen large models:

```bash
python experiments/run_blackwell_semantic_n128.py \
  --models Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct \
  --results_root results_blackwell_qwen_large_semantic_n128
```

Only requested Gemma 4-style IDs:

```bash
python experiments/run_blackwell_semantic_n128.py \
  --models google/gemma-4-12b-it google/gemma-4-26b-a4b-it \
  --results_root results_blackwell_gemma4_semantic_n128
```

Verify the exact Hugging Face model IDs and license gates on the target machine before spending a full run on Gemma 4.

## Returned Files

The runner writes:

```text
results_blackwell_semantic_n128/_run_manifest.json
results_blackwell_semantic_n128/_sweep.log
results_blackwell_semantic_n128/_failures.json
results_blackwell_semantic_n128/all_results.csv
results_blackwell_semantic_n128/<model>/semantic/seed_<seed>/<dataset>/<engine>/benchmark_*.json
```

The most important file for quick inspection is `all_results.csv`.
