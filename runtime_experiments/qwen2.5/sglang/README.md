# SGLang and LMCache Runtime Results

This folder contains the curated SGLang table and the included Qwen2.5-32B SGLang/LMCache run files.

## Files

| Path | Contents |
|---|---|
| `results.csv` | Curated table for SGLang RadixAttention, RadixAttention + MeritKV, and LMCache. |
| `raw/q32b_full_20260608/` | Full Qwen2.5-32B SGLang/LMCache run tree, including summary CSV/JSON/MD files and stdout/stderr logs. |

## Notes

- `results.csv` is the public-facing table.
- The `raw/` folder keeps the run files inside the SGLang runtime folder.
- LMCache also has a curated subset in `../lmcache/results.csv`.

