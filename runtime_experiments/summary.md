# Cross-Runtime Summary

## SGLang: ShadowKV++ Speedup vs LMCache Baseline

Mean across the represented datasets and modes:

| Model | ShadowKV++ vs LMCache |
|-------|----------------------|
| Qwen2.5-1.54B | +7.2% |
| Qwen2.5-3.09B | +8.4% |
| Qwen2.5-7.61B | +16.7% |
| Qwen2.5-14.7B | +12.7% |
| Qwen2.5-32.5B | +5.1% |

## SGLang: ShadowKV++ Speedup vs Native RadixAttention

Mean across the represented datasets and modes:

| Model | ShadowKV++ vs RadixAttention |
|-------|----------------------------|
| Qwen2.5-1.54B | -1.1% |
| Qwen2.5-3.09B | -0.8% |
| Qwen2.5-7.61B | +2.9% |
| Qwen2.5-14.7B | +1.7% |
| Qwen2.5-32.5B | +3.7% |

## vLLM: Speedup vs No-Cache Baseline

Mean across the represented 32B rows:

| Engine | vs No Cache |
|--------|------------|
| vllm_no_cache | +0.0% |
| vllm_apc | +17.9% |
| vllm_apc_shadowkv_plus | +18.5% |

## Data Volume

| Table | Rows | Models | Engines | Dataset Coverage |
|-------|-----:|-------:|--------:|------------------|
| SGLang | 290 | 5 | 3 | Ten-dataset table with omitted/derived cells noted in `sglang/README.md` |
| vLLM | 270 | 5 | 3 | Table includes measured 32B anchors and scaled/projected rows noted in `vllm/README.md` |
| LMCache | 100 | 5 | 1 | Complete 5 x 1 x 10 x 2 table |
