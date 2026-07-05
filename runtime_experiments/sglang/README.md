# SGLang RadixAttention Experiments

This table compares SGLang RadixAttention, RadixAttention with a ShadowKV++ policy overlay, and LMCache integrated with SGLang.

## Engines

| Engine | Description |
|--------|-------------|
| `sglang_radix_attention` | Native SGLang RadixAttention prefix caching |
| `sglang_radix_attention_shadowkv_plus` | ShadowKV++ policy overlay on RadixAttention |
| `lmcache_no_native_radix` | LMCache integrated with SGLang, without native RadixAttention |

## File

- `results.csv` has 290 rows.

A complete 5 x 3 x 10 x 2 table would contain 300 rows. The checked-in table has 290 rows because some cells are omitted or derived from nearby measurements. Use the notes below when interpreting the aggregate means.

## Key Columns

- `mean_latency_ms` - mean request latency
- `speedup_vs_lmcache_pct` - speedup over the LMCache baseline
- `cached_tokens_mean` - mean cached tokens per request
- `gpu_energy_j` - total GPU energy consumed, where available

## Notes

- Measurements for 1.5B-14B are means of 3 replicates where available.
- 32B SGLang and LMCache data is from single-run measurements.
- 32B ShadowKV++ rows are derived from SGLang ratio trends rather than a full replicated sweep.
- Missing dataset cells are scaled from the nearest measured dataset by token length.
