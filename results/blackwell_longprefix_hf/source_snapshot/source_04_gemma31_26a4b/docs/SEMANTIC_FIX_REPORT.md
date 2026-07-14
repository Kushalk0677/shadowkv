# Semantic Opportunity Fix Report

## Problem observed

The previous semantic-mode HF run produced useful adaptive-controller results, but all semantic novelty counters stayed at zero:

- `policy_semantic_partial_total = 0`
- `semantic_opportunity_plans_total = 0`
- `semantic_blocked_by_backend_total = 0`

Root cause: the semantic index observed the current request before planning. The current scaffold could then win as a self-match and be relabeled as an exact-prefix signal, even when no reusable cached KV state existed. As a result, semantic opportunities were hidden behind exact/bypass decisions.

## Fixes made

1. **Semantic self-match suppression**
   - `ShadowKVPlusEngine._semantic_best_match()` now excludes the current just-observed scaffold unless it is already present in the real cache bank.
   - This prevents semantic-mode requests from being counted as their own neighbours.

2. **Semantic vs exact separation**
   - Semantic neighbour LCP is no longer converted into an exact-prefix plan in `prompt_mode='semantic'`.
   - Exact reuse must come from `bank.peek_match()`; semantic opportunities remain semantic opportunities.

3. **Semantic opportunity utility tuning**
   - `AdaptiveReuseController` now treats explicit semantic prompt mode as a reviewable opportunity-measurement path.
   - The semantic branch uses a less pessimistic benefit/waste model when semantic equivalence metadata exists.

4. **Safe HF behaviour preserved**
   - Approximate semantic KV reuse remains blocked on real/HF backends by default.
   - The engine records opportunity and `semantic_blocked_by_backend_total`, then safely performs full prefill without paying unnecessary exact-prefix store cost.

## Smoke validation

A direct fake-backend check with approximate semantic reuse disabled produced:

```text
policy_semantic_partial_total      = 7
semantic_opportunity_plans_total   = 7
semantic_blocked_by_backend_total  = 7
store_attempts                     = 0
```

This validates that semantic opportunity metrics are now alive and safely blocked when approximate reuse is disabled.

## What to run next

For a quick semantic public-dataset HF run:

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
  --include_experimental \
  --output_dir results_semantic_fix_qwen_ag_news
```

Expected for `shadow_kv_plus` on HF semantic mode:

- `semantic_opportunity_plans_total > 0`
- `semantic_blocked_by_backend_total > 0`
- `semantic_partial_hits = 0` unless approximate reuse is explicitly enabled for a safe ablation backend.
