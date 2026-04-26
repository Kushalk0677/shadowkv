# Raw-mode Mitigation Variants

This patch replaces the failed short-prefix raw conservative gate with a stricter raw utility gate and adds two explicit baselines so raw-mode mitigation can be evaluated side-by-side.

## Engines added / changed

### `shadow_kv_plus` — main method, strict raw utility gate

Raw prompts now start in no-store/no-speculation mode. ShadowKV++ only graduates into reuse if all of the following hold:

- at least 12 raw requests have been observed,
- a repeated prefix is at least 48 tokens,
- prefix observation count is at least 8,
- observed frequency is at least 0.35,
- estimated net benefit is at least 8 ms.

This is intentionally conservative because the previous raw gate graduated after short prefixes and caused raw degradation.

### `shadow_kv_plus_raw_observer` — raw-safe no-store baseline

This baseline observes raw prompts but never stores, speculates, or semantically queries raw requests. It is intended to approximate the safest fallback behavior for raw workloads.

### `shadow_kv_plus_best_latency` — previous fastpath baseline

This preserves the earlier best-latency fastpath behavior before the stricter raw-conservative gate. It lets the paper compare the new strict gate against the best prior implementation rather than relying on memory of older runs.

## New / updated metrics

- `raw_strategy`
- `raw_observer_bypass_total`
- `raw_conservative_bypass_total`
- `raw_reuse_evidence_weak_total`
- `raw_reuse_evidence_strong_total`
- `raw_graduated_total`
- `raw_graduate_min_observations`
- `raw_graduate_min_frequency`
- `raw_graduate_min_prefix_len`

## Expected evaluation logic

For raw mode, compare:

1. `shadow_kv_plus`
2. `shadow_kv_plus_raw_observer`
3. `shadow_kv_plus_best_latency`
4. `shadow_kv`
5. `strict_reactive_prefix_cache`
6. `no_cache`

The best final strategy is the one that avoids raw degradation while preserving templated and semantic gains.
