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
