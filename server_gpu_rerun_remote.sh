#!/usr/bin/env bash
set -euo pipefail

cd ~/research
rm -rf shadowkv
git clone https://github.com/Kushalk0677/shadowkv.git
cd ~/research/shadowkv

mkdir -p results

cat > run_all_gpu.sh <<'EOF'
for d in ag_news alpaca_eval banking77 cnn_dailymail daily_dialog dolly oasst1 samsum ultrachat xsum; do
  echo "=== $d ==="
  python experiments/run_benchmark.py \
    --backend hf \
    --model qwen25_15b \
    --device cuda:0 \
    --dtype float16 \
    --workload public_dataset \
    --dataset "$d" \
    --n_requests 16 \
    --include_experimental \
    --output_dir results/ > "results/run_${d}.log" 2>&1 || echo "FAILED: $d"
done
EOF

bash run_all_gpu.sh
