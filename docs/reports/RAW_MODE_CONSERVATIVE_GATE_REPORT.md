# Raw-Mode Conservative Gate Patch

This patch makes ShadowKV++ more conservative on raw prompt workloads.

## Problem
Raw workloads often do not contain stable reusable scaffolds. The previous fastpath activated only after observing low reuse density, meaning early raw requests could still pay policy/store/speculation overhead before the system learned that reuse was weak.

## Change
ShadowKV++ now starts raw mode in a no-store/no-spec conservative path. It observes prefix frequencies cheaply using the existing query observation bank and graduates into normal cache planning only when there is strong evidence of repeated prefixes.

## New behavior
For `prompt_mode=raw` and safe ShadowKV++ mode:

1. Observe the request prefix statistics.
2. If no repeated-prefix evidence is strong enough:
   - run full prefill only,
   - do not store reactive prefixes,
   - do not run semantic matching,
   - push speculation into cooldown,
   - record a fast bypass.
3. If repeated-prefix evidence becomes strong:
   - normal ShadowKV++ exact reuse planning is allowed.

## New metrics

- `raw_conservative_bypass_total`
- `raw_reuse_evidence_strong_total`
- `raw_reuse_evidence_weak_total`

Existing metrics preserved:

- `fast_raw_bypass_total`
- `semantic_queries_skipped_total`

## Graduation thresholds

Configured inside `ShadowKVPlusEngine`:

- `raw_graduate_min_observations = 3`
- `raw_graduate_min_frequency = 0.22`
- `raw_graduate_min_prefix_len = 16`

These are deliberately conservative for N=32 smoke tests. For larger paper-grade runs, tune these thresholds by validation split.

## Expected result

Raw mode should no longer suffer from early reactive-store/speculation overhead on weak datasets such as `oasst1`, `dolly`, and `banking77`. Templated and semantic modes should remain unchanged.
