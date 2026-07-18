# Gemma 4 Runtime Matrix Audit

## Verification

- Result cells: 150/150
- Unique expected cells: 150/150
- Requests: 38,400 total; all cells at 256: True
- NVML energy present without errors: True
- Positive runtime-native cache evidence for every model/runtime pair: True
- Individual zero-hit cells: 10
- Reuse failures: 0
- Idle stabilization timeouts: 0
- LMCache logs with both stores and retrieves: 5/5

## Overall Runtime Means

| Runtime | Mean E2E latency (ms) | Mean P95 (ms) | Mean throughput (req/s) | Mean GPU energy/cell (J) | Cache evidence tokens |
|---|---:|---:|---:|---:|---:|
| vllm_apc | 25.93 | 32.43 | 60.43 | 3344.96 | 1,448,960 |
| sglang_radix_attention | 47.51 | 54.97 | 28.09 | 5496.17 | 1,629,265 |
| lmcache_vllm | 31.97 | 39.70 | 47.96 | 4472.60 | 66,560 |

## Matched Comparisons Against vLLM APC

- `lmcache_vllm`: mean latency +24.6%, P95 +24.2%, GPU energy +39.5% vs vLLM APC; latency wins 0/50 matched cells.
- `sglang_radix_attention`: mean latency +100.6%, P95 +86.2%, GPU energy +70.2% vs vLLM APC; latency wins 0/50 matched cells.

## Caveats And Anomalies

- The three systems use different runtime builds. The version files and image inspection records are included under `metadata/`.
- `cached_tokens_total` is not authoritative for vLLM/LMCache in this harness. vLLM local/external Prometheus counter deltas and LMCache transfer logs are used instead.
- SGLang and vLLM tokenize/format requests through their own server paths; prompt-token totals can differ, so this is a system-level runtime comparison rather than a kernel-only comparison.
- One randomized block order was used. This matrix has one run per cell, so small differences should not be treated as replicated performance estimates.
- LMCache uses an external CPU KV tier with 256-token chunks; short reusable prefixes below that granularity do not produce external hits.
- Zero-hit cells are concentrated in `lmcache_vllm/ag_news: 10`. The raw requests completed successfully; these cells measured runtime overhead without an external KV hit.
- All cells reached the configured idle-power stabilization criterion before energy measurement.

## Audit Result

**PASS: coverage, request counts, energy capture, cache evidence, and failure checks all passed.**
