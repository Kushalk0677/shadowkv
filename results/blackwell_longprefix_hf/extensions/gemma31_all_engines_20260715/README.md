# Gemma 4 31B All-Engine Extension

## Engine Name Aliases

Raw artifacts keep the stable engine IDs used during execution. Public-facing text maps them as follows:

| Engine ID | Display name |
|---|---|
| `shadow_kv_plus` | MeritKV |
| `shadow_kv` | MeritKV-Sem |
| `shadow_kv_plus_lite` | MeritKV-Lite |


This extension adds the `ShadowKV_HF_Blackwell_Gemma4-31B_AllEngines_20260715.zip` artifact to the Blackwell long-prefix HF result package without merging it into the main 5-seed combined tables.

## Why Separate

The main folder summarizes the kept long-prefix HF model set around the MeritKV (`shadow_kv_plus`) engine. This extension is a different slice: one Gemma 4 31B model, one seed, semantic long-prefix mode, and all available HF engines. Keeping it separate avoids mixing a single-seed all-engine sweep into the main multi-seed model summaries.

## Scope

- Model: `google/gemma-4-31B-it`
- Backend: Hugging Face external-KV path on NVIDIA RTX PRO 6000 Blackwell
- Mode: semantic long-prefix scaffold, `n_requests=128`
- Seed: `42`
- Datasets: 10 public datasets
- Engines: 11 engines, including no-cache, reactive/greedy/strict reactive, MeritKV-Sem (`shadow_kv`), and four MeritKV-family (`shadow_kv_plus*`) variants
- Raw coverage: 110 benchmark JSONs, 110 run manifests, and 110 policy traces

## Included Files

- `ShadowKV_HF_Blackwell_Gemma4-31B_AllEngines_20260715.zip`: original source archive.
- `SOURCE_ARCHIVE_SHA256.txt`: checksum for the source archive.
- `artifact/gemma31_all_engines_longprefix_n128_20260714/`: extracted artifact contents, including README, analysis CSVs, raw results, smoke results, runner scripts, metadata, and logs.

## Result Summary

The source README reports that all 110 full cells and all 11 smoke cells completed successfully. The MeritKV (`shadow_kv_plus`)-family engines show about 31% aggregate mean-latency improvement, about 37% P95 improvement, and about 33% GPU-energy improvement versus no-cache, with 1,270/1,280 reuse successes.

The important interpretation is that this artifact supports exact long-scaffold reuse on Gemma 4 31B under the HF backend. It should not be presented as evidence for semantic-partial reuse: the policy counters show zero semantic partial hits in this run.

## Caveats

- Single seed and fixed engine order.
- The `native_prefix_cache` row is an HF placeholder/observer, not a real vLLM APC or SGLang Radix runtime result.
- The source artifact keeps historical labels and stable engine IDs; this wrapper maps `shadow_kv_plus` to MeritKV, `shadow_kv` to MeritKV-Sem, and `shadow_kv_plus_lite` to MeritKV-Lite.
