# SGLang RadixAttention Experiments

This table compares SGLang RadixAttention, RadixAttention with a MeritKV policy overlay, and LMCache integrated with SGLang.

## Engines

| Engine | Description |
|--------|-------------|
| `sglang_radix_attention` | Native SGLang RadixAttention prefix caching |
| `sglang_radix_attention_shadowkv_plus` | MeritKV policy overlay on RadixAttention |
| `lmcache_no_native_radix` | LMCache integrated with SGLang, without native RadixAttention |

## File

- `results.csv` has 290 rows covering 5 model sizes across 10 datasets and 2 prompt modes.

## Key Columns

- `mean_latency_ms` - mean request latency
- `speedup_vs_lmcache_pct` - speedup over the LMCache baseline
- `cached_tokens_mean` - mean cached tokens per request
- `gpu_energy_j` - total GPU energy, where available

## Notes

- Measurements were prepared on an NVIDIA RTX PRO 6000 Blackwell environment.
- All table values are per-engine means across 3 independent runs where available.
