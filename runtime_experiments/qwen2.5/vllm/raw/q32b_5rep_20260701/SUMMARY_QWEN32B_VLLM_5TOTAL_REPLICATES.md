# Qwen32B vLLM 5-Total-Replicate Summary

Rows: `150` / expected `150`
Reps present: `[1, 2, 3, 4, 5]`

## Cached-Token Observability

The cached-token values in this summary are from vLLM Prometheus metrics, not from the OpenAI-compatible response usage object. In this vLLM build, the `/v1/completions` responses do not populate `usage.prompt_tokens_details.cached_tokens`, so response-level cached-token fields can remain `0`/null even when APC is working.

Use these metric-backed fields as the cache-hit evidence:

- `vllm_prompt_tokens_cached_delta`
- `vllm_prefix_cache_hits_delta`
- `vllm_local_cache_hit_tokens_delta`

## By Baseline

| Baseline | Jobs | Mean ms | P95 ms | RPS | Idle J/req | Metrics cached tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vllm_apc | 50 | 59.37 | 73.62 | 17.25 | 26.63 | 1472032 |
| vllm_apc_shadowkv_plus | 50 | 59.62 | 73.93 | 17.17 | 26.59 | 1470400 |
| vllm_no_cache | 50 | 72.84 | 89.96 | 14.21 | 36.32 | 0 |

## Paired Deltas

| Comparison | Cells | Mean latency delta | P95 delta | Throughput delta | Idle J/req delta | Cached-token delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vllm_apc_shadowkv_plus_vs_vllm_apc | 50 | 0.44% | 0.59% | -0.42% | -0.22% | -1632 |
| vllm_apc_shadowkv_plus_vs_vllm_no_cache | 50 | -17.59% | -18.03% | 21.56% | -27.26% | 1470400 |
| vllm_apc_vs_vllm_no_cache | 50 | -17.95% | -18.42% | 22.10% | -27.09% | 1472032 |

