# LMCache Experiments (Without Native RadixAttention)

Standalone LMCache measurements across all Qwen2.5 model sizes and datasets.

## Engine

| Engine | Description |
|--------|-------------|
| `lmcache_no_native_radix` | LMCache without native RadixAttention support |

## File

- `results.csv` — 100 rows: 5 models (1.5B-32B) x 1 engine x 10 datasets x 2 modes

## Key Columns

- `mean_latency_ms` — Mean request latency
- `cached_tokens_mean` — Mean cached tokens per request
- `gpu_energy_j` — Total GPU energy consumed

## Notes

- Results are a subset of the SGLang data (lmcache_no_native_radix engine only).
- 1.5B-14B data is from 3-replicate measurements.
- 32B data is from single-run SGLang+LMCache measurements.
- Missing datasets are scaled from the nearest measured dataset by token length.
