# Request-level policy trace logging

Every `experiments/run_benchmark.py` run now emits a request-level JSONL file:

```text
<output_dir>/policy_trace.jsonl
```

Each line is one engine/request observation. The trace is designed to turn aggregate benchmark runs into a proper policy-learning dataset rather than a tiny benchmark-level table.

Key columns include:

- `engine`, `request_id`, `model`, `dataset`, `variant`, `prompt_mode`, `seed`
- `latency_ms`, `token_count`, `matched_prefix_length`, `tokens_recomputed`
- `was_cache_hit`, `was_speculative_hit`, `cache_tier`
- cumulative reuse/store/waste counters at the time of the request
- ShadowKV++ policy fields when available:
  - `policy_strategy`
  - `policy_reusable_prefix_tokens`
  - `policy_expected_benefit_ms`
  - `policy_expected_cost_ms`
  - `policy_expected_waste_ms`
  - `policy_score_ms`
  - `policy_confidence`
  - `policy_reason`
- semantic fields when available:
  - `semantic_similarity`
  - `semantic_prefix_len`
  - `semantic_lcp_len`
  - `semantic_match_available`
- bounded metadata such as `semantic_equivalence_key`, `semantic_family`, and `shared_prefix_hint_tokens`.

A benchmark with 7 engines and 512 requests should produce about 3584 trace rows. This is the basis for training and validating a learned policy from request-level or candidate-level observations instead of relying on a small benchmark-level training set.
