# Cross-Runtime Summary

## SGLang: ShadowKV++ Speedup vs LMCache Baseline

Mean across all 10 datasets, both modes:

| Model | ShadowKV++ vs LMCache |
|-------|----------------------|
| Qwen2.5-1.54B | +7.2% |
| Qwen2.5-3.09B | +8.4% |
| Qwen2.5-7.61B | +15.7% |
| Qwen2.5-14.7B | +12.7% |
| Qwen2.5-32.5B | +2.7% |

## SGLang: ShadowKV++ Speedup vs Native RadixAttention

Mean across all 10 datasets, both modes:

| Model | ShadowKV++ vs RadixAttention |
|-------|----------------------------|
| Qwen2.5-1.54B | -1.1% |
| Qwen2.5-3.09B | -0.8% |
| Qwen2.5-7.61B | +2.9% |
| Qwen2.5-14.7B | +1.7% |
| Qwen2.5-32.5B | +3.7% |

## vLLM: Speedup vs No-Cache Baseline

Mean across 5 measured datasets, both modes (32B):

| Engine | vs No Cache |
|--------|------------|
| vllm_no_cache | +0.0% |
| vllm_apc | +17.9% |
| vllm_apc_shadowkv_plus | +18.5% |

## Data Volume

| Dataset | Rows | Models | Engines | Datasets |
|---------|------|--------|---------|----------|
| SGLang | 290 | 5 | 3 | 10 |
| vLLM | 270 | 5 | 3 | 10 |
| LMCache | 100 | 5 | 1 | 10 |
