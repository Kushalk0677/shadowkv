# vLLM Runtime Results

This folder contains the curated vLLM table and the included Qwen2.5-32B run files.

## Files

| Path | Contents |
|---|---|
| `results.csv` | Curated vLLM presentation table covering no-cache, APC, and APC + MeritKV. |
| `raw/q32b_5rep_20260701/` | 5-replicate Qwen2.5-32B aggregate, summary, and available benchmark JSONs. |
| `raw/q32b_20260603/` | Earlier full Qwen2.5-32B run tree with no-cache, APC, and APC + MeritKV benchmark JSONs. |

## Notes

- `results.csv` is the public-facing table.
- The `raw/` folders keep the corresponding run files inside the vLLM runtime folder.
- The July 1 aggregate has 150 rows: 5 reps x 10 dataset/mode cells x 3 engines.

