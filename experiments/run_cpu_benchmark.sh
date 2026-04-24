#!/usr/bin/env bash
set -euo pipefail
python experiments/run_benchmark.py \
  --backend hf \
  --model distilgpt2 \
  --device cpu \
  --workload public_dataset \
  --dataset samsum \
  --n_requests 64 \
  --simulate_arrivals \
  --output_dir results/
