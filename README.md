# ShadowKV / ShadowKV++ Research Repository

This is now the integrated full research-grade repository. It preserves the original benchmark harness and result archives, while adding **ShadowKV++** as a policy-driven, semantic-signal, fine-grained, waste-aware KV reuse engine.

## What is new in ShadowKV++

ShadowKV++ changes the research framing from "a cache that reacts" to "an inference controller that decides."

New integrated modules:

- `src/proactive_kv_cache/controller.py` — pre-execution adaptive reuse controller.
- `src/proactive_kv_cache/semantic.py` — lightweight semantic/structural KV prefix index.
- `src/proactive_kv_cache/policy_learning.py` — offline learner for deployment thresholds from benchmark logs.
- `ShadowKVPlusEngine` in `src/proactive_kv_cache/engines.py`.
- `experiments/analyze_shadowkv_results.py` — parses JSON files and result zips into CSV, Markdown, and learned policy JSON.
- `docs/shadowkv_plus_research_design.md` — paper-grade research design and claim boundaries.

Important correctness boundary: real backends only perform exact-prefix KV reuse. Approximate semantic partial reuse is treated as simulator/research-mode unless a backend-specific correctness validation is added.

## Fast validation

```bash
python -m pytest -q
python experiments/run_benchmark.py --backend fake --workload synthetic --variant high_skew --n_requests 40 --include_experimental --disable_arrival_simulation --output_dir results/
python experiments/analyze_shadowkv_results.py results/
```

## Research-grade comparison command

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model distilgpt2 \
  --device cpu \
  --workload public_dataset \
  --dataset samsum \
  --prompt_mode templated \
  --n_requests 64 \
  --include_experimental \
  --output_dir results/
```

---

# ShadowKV: Tiered Prefix Caching for LLM Serving

Research code for testing adaptive prefix reuse in LLM serving. The repository includes CPU-friendly baselines, public dataset loaders, benchmark scripts, and basic diagnostics.

## Included baselines

Default benchmark set:
1. **NoCacheEngine**
   - every request performs full prefill
2. **ReactivePrefixCacheEngine**
   - classic reactive prefix reuse after the first request
3. **StrictReactivePrefixCacheEngine**
   - a more conservative variant with stronger reuse gates

Experimental engines, enabled with `--include_experimental`:
- **FrequencySpeculativeEngine**
- **ShadowKVEngine**

Runtime cache baselines:
- **vLLM APC**
- **vLLM APC + ShadowKV++ admission controller**
- **SGLang RadixAttention**
- **SGLang RadixAttention + ShadowKV++ admission controller**
- **LMCache**
- **LMCache + ShadowKV++ admission controller**

These are additive; the existing baselines and semantic ablations are unchanged.
For literature-accurate results, run runtime-system baselines through
`literature_accurate_baselines/run_runtime_cache_baseline.py`. The main
`experiments/run_benchmark.py --include_runtime_baselines` path only adds vLLM
APC variants when `--backend vllm`, because SGLang RadixAttention and LMCache
must be measured against their actual external runtimes.

## Supported public datasets

Conversational:
- `daily_dialog` -> DailyDialog-style chat prompts
- `oasst1` -> OpenAssistant conversational/instructional turns

Instruction:
- `dolly` -> Databricks Dolly 15k instruction data
- `alpaca_eval` -> AlpacaEval instruction prompts

Also included:
- `xsum`
- `cnn_dailymail`
- `ag_news`
- `banking77`
- `ultrachat`

Some Hugging Face datasets use script-based loaders that are not supported by recent `datasets` releases. The registry uses compatible parquet mirrors where needed.

## Supported models

Small / CPU-friendly:
- `tiny` -> `sshleifer/tiny-gpt2`
- `distilgpt2` -> `distilgpt2`

Larger / GPU-oriented:
- `mistral7b` -> `mistralai/Mistral-7B-Instruct-v0.3`
- `llama31_8b` -> `meta-llama/Llama-3.1-8B-Instruct`

Notes:
- Mistral and Llama variants may require accepting Hugging Face terms or using a token depending on the model page and license gating.
- `distilgpt2` is a public small GPT-2 variant suitable for lightweight experiments.

## Current status

This is exploratory research code. Fake-backend results are useful for smoke tests, but not for performance claims. Stronger claims need repeated HF or vLLM runs, multiple seeds, and reporting of speculative waste.

For Hugging Face external-KV experiments, check that cached prefix + suffix prefill matches full prefill for the target model family before treating latency numbers as evidence.

## Metrics tracked

Per run:
- mean latency
- p50 latency
- p95 latency
- throughput (requests/sec)
- service throughput (requests/sec based on summed request latency)
- cache hit rate
- speculative hit rate
- speculative precompute count
- speculative cost (ms)
- wasted precomputes
- wasted compute (ms)
- wasted compute ratio
- utility proxy
- mean GPU utilization when available
- promotions / demotions between tiers

## Install

```bash
pip install -r requirements.txt
```

Optional GPU utilization sampling uses `pynvml` through `nvidia-ml-py3` when available.

## Quick smoke test

```bash
python -m pytest -q
python experiments/smoke_test.py
```

## CPU benchmark example

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model distilgpt2 \
  --device cpu \
  --workload public_dataset \
  --dataset samsum \
  --n_requests 64 \
  --simulate_arrivals \
  --output_dir results/
```

## Runtime Baseline Examples

```bash
python experiments/run_benchmark.py \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda:0 \
  --workload public_dataset \
  --dataset samsum \
  --prompt_mode templated \
  --n_requests 64 \
  --include_runtime_baselines \
  --output_dir results/
```

For the full literature-accurate runtime set:

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

## GPU benchmark example

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model mistral7b \
  --device cuda:0 \
  --dtype float16 \
  --workload public_dataset \
  --dataset dolly \
  --n_requests 64 \
  --simulate_arrivals \
  --output_dir results/
```

## Semantic/paraphrase novelty workload

The repository now supports a fourth prompt mode:

```bash
--prompt_mode semantic
```

This mode rotates paraphrased scaffolds for the same underlying task family. It is
intended to evaluate whether ShadowKV++ can detect semantic reuse opportunities
that exact-prefix caches cannot exploit.

For controlled local ablation without downloading datasets:

```bash
python experiments/run_benchmark.py \
  --backend fake \
  --workload synthetic \
  --variant semantic_paraphrase \
  --n_requests 32 \
  --include_experimental \
  --output_dir results/semantic_fake_smoke
```

For HF public-dataset novelty evaluation:

```bash
python experiments/run_semantic_novelty_matrix.py
```

On real HF backends, semantic opportunities are reported but approximate KV reuse
is blocked by default for correctness. Inspect `semantic_opportunity_*` and
`semantic_blocked_by_backend_total` metrics.
