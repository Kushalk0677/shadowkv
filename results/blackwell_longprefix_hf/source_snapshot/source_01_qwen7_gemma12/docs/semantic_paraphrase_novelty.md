# Semantic/paraphrase workload for ShadowKV++ novelty

## Why this workload exists

Raw prompts test safety. Templated prompts test exact structural reuse. Neither
fully demonstrates the new ShadowKV++ thesis: an inference controller should be
able to identify semantically equivalent request families even when the surface
prefix is not identical.

The new `semantic` prompt mode creates paraphrased serving scaffolds for the same
underlying task family. For example, classification requests rotate between
scaffolds such as:

- `System: Classify the item by its dominant intent...`
- `Classification task: identify the category...`
- `Decision brief: read the item...`
- `Task: choose the strongest topic...`

These prompts express the same serving operation but do not share an exact token
prefix. This makes the workload a direct test of whether ShadowKV++ provides
value beyond ordinary prefix caching.

## How it is implemented

`src/proactive_kv_cache/datasets.py` now supports:

```text
--prompt_mode semantic
```

Each semantic prompt carries metadata:

- `semantic_equivalence_key`
- `semantic_family`
- `paraphrase_variant`
- `semantic_variant_count`
- `semantic_anchor_text`

`SemanticKVIndex` accepts an optional semantic equivalence key and boosts matches
inside the same equivalence family. This keeps the implementation lightweight and
reproducible without adding a sentence-transformer dependency to the hot path.

## Correctness boundary

For real HF/vLLM-style backends, ShadowKV++ does **not** blindly reuse approximate
semantic KV tensors because that could change model semantics. Instead, it records
semantic opportunity metrics:

- `policy_semantic_partial_total`
- `semantic_opportunity_plans_total`
- `semantic_opportunity_reused_tokens_total`
- `semantic_opportunity_estimated_savings_ms`
- `semantic_blocked_by_backend_total`

On `FakeBackend`, approximate semantic partial reuse may execute as an ablation.
This gives two separate claims:

1. Real backend: policy and opportunity detection are measured safely.
2. Fake backend: approximate partial reuse potential can be stress-tested.

## Expected paper claim

Do not claim that semantic KV reuse is production-safe yet. The correct claim is:

> ShadowKV++ identifies semantic reuse opportunities that exact-prefix caches miss,
> and exposes a correctness-aware boundary between safe exact reuse and approximate
> semantic reuse opportunities.

This is stronger and more credible than pretending semantic KV reuse is already
universally safe.

## Run command

```bash
python experiments/run_semantic_novelty_matrix.py
```

Or manually:

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --dtype float16 \
  --workload public_dataset \
  --prompt_mode semantic \
  --dataset ag_news \
  --n_requests 32 \
  --mean_inter_arrival_ms 50 \
  --max_arrival_sleep_ms 500 \
  --seed 42 \
  --include_experimental \
  --output_dir results_semantic_novelty/Qwen_Qwen2.5-1.5B-Instruct/seed_42/ag_news
```

## How to evaluate

Compare `semantic` mode against `raw` and `templated`:

- Exact prefix baselines should not dominate simply from repeated identical scaffolds.
- ShadowKV++ should report non-zero semantic opportunity metrics.
- If `semantic_blocked_by_backend_total` is non-zero, that is expected on HF; it means
  the controller found semantic opportunity but avoided unsafe approximate KV reuse.
- On FakeBackend, `semantic_partial_hits` can be non-zero for the ablation.
