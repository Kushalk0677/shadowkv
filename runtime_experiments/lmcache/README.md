# LMCache Experiments Without Native RadixAttention

This table contains standalone LMCache measurements across the represented Qwen2.5 model sizes and datasets.

## Engine

| Engine | Description |
|--------|-------------|
| `lmcache_no_native_radix` | LMCache without native RadixAttention support |

## File

- `results.csv` has 100 rows: 5 models x 1 engine x 10 datasets x 2 modes.

## Key Columns

- `mean_latency_ms` - mean request latency
- `cached_tokens_mean` - mean cached tokens per request
- `gpu_energy_j` - total GPU energy consumed, where available

## Notes

- Results are the LMCache subset used for the SGLang comparison.
- 1.5B-14B rows are from 3-replicate measurements where available.
- 32B rows are from single-run SGLang + LMCache measurements.
- Missing dataset values, if any, are scaled from the nearest measured dataset by token length.
