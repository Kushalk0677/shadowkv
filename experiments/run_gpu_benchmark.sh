#!/usr/bin/env bash
set -euo pipefail

# Hugging Face run using the custom KV path.
python experiments/run_benchmark.py \
  --backend hf \
  --model qwen25_3b \
  --device cuda:0 \
  --dtype float16 \
  --workload synthetic \
  --variant long_shared_prefix \
  --n_requests 96 \
  --simulate_arrivals \
  --output_dir results/

# vLLM run using its own prefix cache.
python experiments/run_benchmark.py \
  --backend vllm \
  --model qwen25_3b \
  --device cuda:0 \
  --dtype float16 \
  --tensor_parallel_size 1 \
  --workload synthetic \
  --variant rag_long_context \
  --n_requests 96 \
  --simulate_arrivals \
  --output_dir results/
