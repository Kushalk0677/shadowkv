# Runtime Experiment Methodology

This document describes the methodology used for the real-world runtime
experiments in `runtime_experiments/`.

## Hardware

- **GPU**: NVIDIA RTX PRO 6000 Blackwell
- **CPU**: Intel Xeon Gold 5418Y
- **RAM**: 128 GB
- **OS**: Linux (Ubuntu 22.04)

## Software

| System | Version |
|--------|---------|
| SGLang | v0.4.0.post2 |
| LMCache | v0.2.1 |
| vLLM | v0.6.0 |
| Python | 3.10 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## Models

All experiments use the Qwen2.5 model family:

| Model | Parameters |
|-------|-----------|
| Qwen2.5-1.5B-Instruct | 1.54B |
| Qwen2.5-3B-Instruct | 3.09B |
| Qwen2.5-7B-Instruct | 7.61B |
| Qwen2.5-14B-Instruct | 14.7B |
| Qwen2.5-32B-Instruct | 32.5B |

## Datasets

Ten datasets from the HuggingFace hub, covering four task types:

| Task Type | Datasets |
|-----------|----------|
| Classification | AG News, Banking77 |
| Instruction | AlpacaEval, Dolly |
| Conversational | DailyDialog, OASST1, UltraChat |
| Summarisation | SAMSum, XSum, CNN/DailyMail |

Two prompt modes: `templated` (shared serving scaffold) and `rag`
(retrieval-augmented generation scaffold). Each with 256 requests per job.

## Engines

### SGLang experiments

Three engines, compared pairwise on identical request sequences:

1. **`sglang_radix_attention`**: Native SGLang RadixAttention prefix caching.
2. **`sglang_radix_attention_shadowkv_plus`**: ShadowKV++ policy controller
   as an overlay on SGLang's RadixAttention.
3. **`lmcache_no_native_radix`**: LMCache without native RadixAttention
   integration (baseline without prefix-tree matching).

All SGLang results use **3 independent replicates** per cell. Reported
values are means across replicates with 95% CIs.

### vLLM experiments

Three engines on Qwen2.5-32B:

1. **`vllm_no_cache`**: vLLM with prefix caching disabled.
2. **`vllm_apc`**: vLLM with Automatic Prefix Caching (hash-based).
3. **`vllm_apc_shadowkv_plus`**: ShadowKV++ overlay on vLLM APC.

vLLM results are single runs. GPU energy measured via NVML.

## Metrics

- **Mean latency**: Average request latency in milliseconds.
- **Throughput**: Requests per second.
- **Cached tokens**: Mean number of KV cache tokens reused per request.
- **GPU energy**: Total GPU energy consumption in Joules (NVML).
- **Speedup**: `(baseline_latency / engine_latency - 1) * 100`.

## Data Sources

Raw benchmark JSON files and run logs are archived in the deliverable
bundles under `runtime_experiments/`. Each bundle includes:

- Individual benchmark JSONs (one per engine/dataset/mode)
- Standard output and error logs
- Run shell scripts for reproducibility
- Hardware metadata (nvidia-smi, pip freeze, Docker image)

## Limitations

- SGLang + ShadowKV++ at 32B is calibrated from 1.5B-14B trend with
  vLLM 32B cross-validation (not directly measured).
- vLLM results for 1.5B-14B are projected from the 32B baseline.
- Results on other model families (GPT-2, TinyLlama, Gemma, Phi-3)
  and other GPU types are not yet available.
