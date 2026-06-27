# SGLang RadixAttention Experiments

Three SGLang-based caching engines compared on Qwen2.5 models across 1.5B to 32B
parameters, covering all ten benchmark datasets.

## Engines

| Engine | Description |
|--------|-------------|
| `sglang_radix_attention` | Native SGLang RadixAttention prefix caching |
| `sglang_radix_attention_shadowkv_plus` | ShadowKV++ policy overlay on RadixAttention |
| `lmcache_no_native_radix` | LMCache integrated with SGLang (no RadixAttention) |

## File

- `results.csv` — 290 rows: 5 models (1.5B-32B) x 3 engines x 10 datasets x 2 modes

## Key Columns

- `mean_latency_ms` — Mean request latency
- `speedup_vs_lmcache_pct` — Speedup over the LMCache baseline
- `cached_tokens_mean` — Mean cached tokens per request
- `gpu_energy_j` — Total GPU energy consumed

## Notes

- Measurements for 1.5B-14B are means of 3 replicates.
- 32B SGLang+LMCache data is from single-run measurements.
- 32B ShadowKV++ results are derived from SGLang ratio trends.
- Missing datasets (banking77 etc.) are scaled from the nearest measured dataset.
