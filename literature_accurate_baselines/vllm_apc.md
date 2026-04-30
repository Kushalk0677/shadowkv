# vLLM APC

Literature/runtime-accurate baseline: vLLM Automatic Prefix Caching (APC).

Primary source: https://docs.vllm.ai/en/v0.18.0/features/automatic_prefix_caching/

Accuracy boundary:
- APC is a vLLM runtime feature, not a heuristic implemented inside this repo.
- The accurate launch path uses `vllm serve ... --enable-prefix-caching`.
- APC reuses exact shared-prefix KV blocks and primarily improves prefill/TTFT,
  not long decode-heavy generations.

Run:

```bash
python literature_accurate_baselines/run_runtime_cache_baseline.py \
  --baseline vllm_apc \
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
  --baseline vllm_apc_shadowkv_plus \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --workload public_dataset \
  --dataset samsum \
  --prompt_mode templated \
  --n_requests 64 \
  --launch_server
```

The admission-controller variant uses vLLM's documented prefix-cache reset
endpoint to conservatively enforce controller bypass decisions.
