# ShadowKV v10

Research prototype for adaptive prefix/KV reuse in LLM serving. The v10
snapshot contains CPU-friendly fake-backend smoke tests, Hugging Face and vLLM
backend adapters, ShadowKV/ShadowKV++ engines, public workload builders, and
external runtime baseline launchers.

## Engine Families

Default benchmark engines:
- `no_cache`
- `native_prefix_cache` for native runtime prefix caching
- `reactive_prefix_cache`
- `greedy_prefix_cache`
- `strict_reactive_prefix_cache`

Experimental engines enabled with `--include_experimental`:
- `frequency_speculative`
- `shadow_kv`
- `shadow_kv_plus`
- `shadow_kv_plus_lite`
- `shadow_kv_plus_best_latency`
- `shadow_kv_plus_raw_observer`

Semantic ablations enabled with `--include_semantic_ablations`:
- `shadow_kv_plus_scaffold_only`
- `shadow_kv_plus_early_layer`
- `shadow_kv_plus_logit_guard`

Runtime-system baselines such as vLLM APC, SGLang RadixAttention, SGLang
HiCache, LMCache, and KVFlow live under `literature_accurate_baselines/`.
Use those adapters for external runtime measurements instead of treating fake
or Hugging Face in-process results as runtime-system reproductions.

## Quick Checks

```bash
python -m pytest -q
python experiments/run_benchmark.py --backend fake --workload synthetic --variant high_skew --n_requests 10 --include_experimental --disable_arrival_simulation --output_dir tmp_shadowkv_v10_smoke
```

## Notes

Fake-backend numbers are only smoke-test evidence. Performance claims need
repeated Hugging Face or runtime-server runs, multiple seeds, and reporting of
hit rate, waste, energy, and cache admission metrics.
