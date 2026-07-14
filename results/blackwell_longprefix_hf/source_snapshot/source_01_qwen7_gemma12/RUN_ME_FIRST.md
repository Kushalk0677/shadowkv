# Run Me First: Blackwell Semantic n=128

This repo bundle is for the RTX PRO 6000 Blackwell semantic-reuse run.

From this repo root, run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Then dry run:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py --dry_run
```

Then smoke test:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py \
  --models gpt2 \
  --datasets ag_news \
  --n_requests 16 \
  --no-measure_energy \
  --results_root results_blackwell_semantic_smoke
```

Then full run:

```bash
python blackwell_semantic_n128/run_blackwell_semantic_n128.py
```

The full run uses semantic mode, seed `42`, `n=128`, and the three isolated
engines:

```text
no_cache
shadow_kv
shadow_kv_plus
```

It also uses longer shared semantic scaffolds by default:

```text
semantic_shared_prefix_repeats = 4
semantic_shared_prefix_mode = common_scaffold
```

After the run, use `comparisons_vs_no_cache.csv` for latency/energy and
`reuse_path_breakdown.csv` to separate exact scaffold reuse from semantic partial
reuse. Per-request `policy_trace.jsonl` files are enabled by default for audit
backup.

Read `BLACKWELL_SEMANTIC_N128_HANDOFF.md` for details, model list, expected
outputs, and optional literature baselines.
