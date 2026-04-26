# ShadowKV++ Full Repository Integration Report

## What changed

This repository now preserves the original benchmark/results structure and adds ShadowKV++ as an integrated experimental engine.

Added:
- `ShadowKVPlusEngine`
- `AdaptiveReuseController`
- `SemanticKVIndex`
- `policy_learning.py`
- `experiments/analyze_shadowkv_results.py`
- `docs/shadowkv_plus_research_design.md`
- result-preserving integration with existing benchmark script

Preserved:
- original public dataset loaders
- original synthetic workloads
- original baseline engines
- original archived T4 result zips
- original tests
- original CPU/GPU benchmark entry points

## How to run

```bash
python -m pytest -q
python experiments/run_benchmark.py --backend fake --workload synthetic --variant high_skew --n_requests 40 --include_experimental --disable_arrival_simulation --output_dir results/
python experiments/analyze_shadowkv_results.py results/
```

## ShadowKV++ engine name

Use:

```bash
--include_experimental
```

The benchmark will include:

```text
frequency_speculative
shadow_kv
shadow_kv_plus
```

## Correctness posture

Real HF/vLLM-style external KV reuse remains exact-prefix only.

Semantic partial reuse is a simulator/research-mode mechanism unless explicitly enabled and validated for a backend.

## Validation performed in this integration

- Python compile check for modified modules.
- Full unit test suite.
- Fake-backend quick benchmark with `shadow_kv_plus` included.
- Existing result archives preserved.


## Local fake-backend validation matrix

These are smoke/regression numbers only, not publishable latency evidence.

| variant | strict reactive | ShadowKV | ShadowKV++ | ++ bypass plans | ++ net utility |
|---|---:|---:|---:|---:|---:|
| high_skew | 1.000 | 1.366 | 1.366 | 2 | 543.4 |
| bursty_high | 1.000 | 1.129 | 1.156 | 5 | 333.3 |
| bursty_mild | 1.000 | 0.853 | 1.019 | 9 | 252.3 |
| long_shared_prefix | 2.679 | 2.170 | 2.552 | 1 | 5215.0 |
| low_skew | 1.000 | 0.862 | 0.940 | 8 | 273.0 |
| mild_skew | 1.000 | 0.843 | 1.007 | 8 | 245.2 |
| moderate_skew | 1.000 | 0.913 | 0.968 | 5 | 290.8 |
| rag_long_context | 2.375 | 1.941 | 2.281 | 1 | 5170.7 |
| uniform | 1.000 | 0.861 | 1.017 | 10 | 386.6 |

Key observation: ShadowKV++ is now materially better than the old ShadowKV on low/medium reuse fake workloads because the policy bypasses negative-utility behavior. On long structured prefixes, strict reactive can still be stronger, which is honestly documented and should be addressed with real backend-level fine-grained KV reuse if the paper wants to claim superiority over all baselines.

## Semantic/paraphrase novelty extension

This integration adds a fourth public prompt mode: `semantic`.

Unlike `templated`, the `semantic` mode deliberately rotates paraphrased scaffolds
for the same serving task. This creates semantically equivalent request families
without identical token prefixes, so it is a direct novelty test against ordinary
prefix caching.

New metrics emitted by `shadow_kv_plus`:

- `policy_semantic_partial_total`
- `semantic_opportunity_plans_total`
- `semantic_opportunity_reused_tokens_total`
- `semantic_opportunity_estimated_savings_ms`
- `semantic_blocked_by_backend_total`

Correctness boundary: on real HF backends, approximate semantic KV reuse is not
executed by default. Instead, ShadowKV++ records the opportunity and blocks unsafe
approximate reuse. On FakeBackend, semantic partial reuse can execute as an
ablation.

New benchmark entry points:

```bash
python experiments/run_semantic_novelty_matrix.py
python experiments/run_benchmark.py --backend fake --workload synthetic --variant semantic_paraphrase --n_requests 32 --include_experimental --output_dir results/semantic_fake_smoke
```
