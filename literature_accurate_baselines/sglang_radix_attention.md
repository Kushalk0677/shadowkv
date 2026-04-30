# SGLang RadixAttention

Literature/runtime-accurate baseline: SGLang's default RadixAttention prefix
cache.

Primary sources:
- https://docs.sglang.io/
- https://docs.sglang.io/docs/advanced_features/server_arguments
- https://docs.sglang.io/docs/basic_usage/native_api

Accuracy boundary:
- RadixAttention is an SGLang runtime feature, not an in-repo cache heuristic.
- The accurate baseline launches/attaches to `sglang.launch_server` without
  `--disable-radix-cache`.
- The adapter rejects `--disable-radix-cache` for this baseline.

Run:

```bash
python literature_accurate_baselines/run_runtime_cache_baseline.py \
  --baseline sglang_radix_attention \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --workload public_dataset \
  --dataset samsum \
  --prompt_mode templated \
  --n_requests 64 \
  --launch_server
```

ShadowKV++ admission-controller variant:

```bash
python literature_accurate_baselines/run_runtime_cache_baseline.py \
  --baseline sglang_radix_attention_shadowkv_plus \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --workload public_dataset \
  --dataset samsum \
  --prompt_mode templated \
  --n_requests 64 \
  --launch_server
```

The admission-controller variant uses SGLang's documented `/flush_cache`
endpoint to conservatively enforce bypass decisions.
