# Semantic Correctness Ablations

ShadowKV++ now exposes three semantic-reuse ablation engines. They are intentionally separate from the default safe `shadow_kv_plus` engine so the paper can distinguish detected semantic opportunity from actually executed approximate KV reuse.

## Engines

Enable them with:

```bash
--include_semantic_ablations
```

This adds:

- `shadow_kv_plus_scaffold_only`
- `shadow_kv_plus_early_layer`
- `shadow_kv_plus_logit_guard`

The default `shadow_kv_plus` remains conservative: it detects semantic opportunities but blocks unsafe approximate reuse on real HF backends.

## 1. Scaffold-only reuse

Purpose: conservative baseline.

This engine only executes semantic/scaffold reuse when the reusable region is also an exact scaffold-compatible prefix. Otherwise it records the opportunity but does not substitute approximate KV.

Useful metrics:

- `scaffold_only_attempts`
- `scaffold_only_hits`
- `semantic_opportunity_plans_total`
- `semantic_blocked_by_backend_total`

## 2. Early-layer reuse

Purpose: speed-quality tradeoff curve.

The engine reuses only a configured fraction of the semantic candidate prefix:

```bash
--early_layer_reuse_ratio 0.25
--early_layer_reuse_ratio 0.50
--early_layer_reuse_ratio 0.75
```

This is a controlled ablation. On fake/controlled backends it can execute; on real HF it should be treated as experimental.

Useful metrics:

- `early_layer_attempts`
- `early_layer_hits`
- `early_layer_reuse_ratio_sum`
- `semantic_quality_divergence_sum`
- `semantic_quality_divergence_events`

## 3. Logit-guarded reuse

Purpose: correctness-preserving full semantic reuse.

Before admitting semantic KV reuse, the backend compares next-token distributions after the current scaffold and the candidate semantic scaffold. Reuse is admitted only if the distance is below:

```bash
--logit_guard_threshold 0.08
```

Useful metrics:

- `logit_guard_checks`
- `logit_guard_passes`
- `logit_guard_failures`
- `logit_guard_distance_sum`
- `semantic_guarded_hits`
- `semantic_partial_hits`

## Recommended paper experiment

Run semantic mode on real public datasets:

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --dtype float16 \
  --workload public_dataset \
  --prompt_mode semantic \
  --dataset samsum \
  --n_requests 128 \
  --seed 42 \
  --include_experimental \
  --include_semantic_ablations \
  --output_dir results_semantic_ablations/qwen/samsum
```

For a full quality/speed curve, sweep:

- `--early_layer_reuse_ratio` over `0.25, 0.50, 0.75`
- `--logit_guard_threshold` over `0.04, 0.08, 0.12, 0.20`

Always report:

1. mean latency speedup,
2. p95 speedup,
3. semantic hits,
4. guard pass/fail,
5. divergence proxy,
6. task-quality metrics where labels exist.
