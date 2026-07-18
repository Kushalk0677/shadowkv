# Reproducing P100 Runs

Raw artifacts keep stable engine IDs: `shadow_kv_plus` displays as MeritKV, `shadow_kv` displays as MeritKV-Sem, and `shadow_kv_plus_lite` displays as MeritKV-Lite.


Use `experiments/run_p100_isolated_sweep.py` for a conservative public P100 rerun. The local `p100_transfer/` package remains a transfer artifact and is not needed in this repository.

## Why This Runner Is Isolated

The historical P100 package included a private `--share_backend` experiment path. The public `experiments/run_benchmark.py` does not expose that flag, so the public runner uses one subprocess per engine cell instead. This is slower, but it is safer on a 12 GB P100 because each failed cell exits cleanly.

## Smoke Test

```bash
python experiments/run_p100_isolated_sweep.py \
  --models gpt2 \
  --datasets ag_news \
  --prompt_modes raw \
  --engines no_cache shadow_kv_plus  # shadow_kv_plus displays as MeritKV \
  --seeds 42 \
  --n_requests 8 \
  --results_root results_p100_smoke
```

## Main Controlled Run

```bash
python experiments/run_p100_isolated_sweep.py
```

Default matrix:

```text
5 models x 10 datasets x 3 prompt modes x 3 seeds x 3 engines
```

Default engines:

```text
no_cache
shadow_kv       # MeritKV-Sem
shadow_kv_plus  # MeritKV
```

To execute approximate semantic KV reuse in semantic mode, add:

```bash
--allow_unsafe_semantic_kv_reuse
```

Use that flag only for controlled ablations and report it explicitly.

## Returned Files

```text
results_p100_isolated/_run_manifest.json
results_p100_isolated/_sweep.log
results_p100_isolated/_failures.json
results_p100_isolated/<model>/<mode>/seed_<seed>/<dataset>/<engine>/benchmark_*.json
```
