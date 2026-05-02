# ShadowKV++

ShadowKV++ is a research prototype for **waste-aware, correctness-bounded KV cache reuse** in LLM serving. It extends a tiered prefix-cache benchmark harness with a per-request policy controller that decides when to reuse, when to speculate, and when to bypass the cache entirely.

This repository is prepared for artifact review and reproduction. It contains source code, tests, benchmark scripts, and the two canonical 3-seed hardware result trees used by the current paper draft.

## Included Artifact

Canonical result roots:

- `results/final_p100`
- `results/final_t4`

Generated summaries:

- `results/RESULTS.md`
- `results/manifest.json`
- `results/summary_by_engine.csv`
- `results/summary_by_mode_engine.csv`

Included result inventory:

- 898 benchmark JSON files
- 8532 engine-result rows
- Seeds: `42`, `123`, `456`
- Models: GPT-2, TinyLlama-1.1B-Chat, Qwen2.5-1.5B-Instruct, Gemma-2B-IT, Phi-3-mini-4k-instruct
- Datasets: AG News, Banking77, AlpacaEval, Dolly, DailyDialog, OASST1, UltraChat, SAMSum, XSum, CNN/DailyMail
- Prompt modes: `raw`, `templated`, `semantic`

Note: the T4 result tree has 448 benchmark files rather than 450 because two Phi-3 templated runs were unavailable in the source bundle. This is recorded in `results/manifest.json`.

## Main Claims To Reproduce

From the included JSON files:

| Engine | Mean Speedup | Waste | Hit Rate |
|---|---:|---:|---:|
| `no_cache` | 1.000x | 0.000 | 0.000 |
| `reactive_prefix_cache` | 1.214x | 0.000 | 0.317 |
| `greedy_prefix_cache` | 1.221x | 0.000 | 0.320 |
| `strict_reactive_prefix_cache` | 1.254x | 0.000 | 0.310 |
| `frequency_speculative` | 1.208x | 0.284 | 0.617 |
| `shadow_kv` | 1.287x | 0.264 | 0.606 |
| `shadow_kv_plus` | 1.365x | 0.156 | 0.402 |

Interpretation boundaries:

- Raw-mode ShadowKV++ gains are primarily **bypass/overhead avoidance**, not exact KV reuse.
- Real HF backends should treat approximate semantic KV substitution as unsafe unless an explicit backend guard validates it.
- The included artifact is a controlled benchmark-harness evaluation, not a native vLLM/SGLang/LMCache production integration.

## Repository Layout

```text
src/proactive_kv_cache/          Core engines, cache bank, controller, models
experiments/run_benchmark.py     Main benchmark entry point
experiments/analyze_shadowkv_results.py
                                 Result parser and policy-summary generator
tests/                           Unit and regression tests
docs/                            Design notes and correctness docs
results/final_p100               Canonical P100 result tree
results/final_t4                 Canonical T4 result tree
```

## Setup

Python 3.10+ is recommended.

```bash
git clone <repo-url>
cd <repo>

python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -e .
pip install pytest
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -e .
pip install pytest
```

The project dependencies are declared in `pyproject.toml`; `requirements.txt` is provided for simple environments.

## Validate The Code

Run the test suite:

```bash
python -m pytest -q
```

Expected result for this release:

```text
49 passed, 1 skipped
```

The skipped test is an optional slow Hugging Face KV-correctness check. Enable it with:

```bash
RUN_HF_KV_CORRECTNESS=1 python -m pytest -q tests/test_backend_regressions.py
```

## Regenerate Result Summaries

The included CSV/Markdown summaries can be regenerated from the bundled JSON files:

```bash
python experiments/analyze_shadowkv_results.py \
  results/final_p100 results/final_t4 \
  --csv results/shadowkv_result_summary.csv \
  --markdown results/shadowkv_result_summary.md \
  --policy-json results/shadowkv_plus_learned_policy.json
```

The release also includes a compact handoff summary:

```bash
cat results/RESULTS.md
cat results/manifest.json
```

## Smoke Benchmark

A fast dependency-light smoke test uses the fake backend:

```bash
python experiments/run_benchmark.py \
  --backend fake \
  --workload synthetic \
  --variant high_skew \
  --n_requests 40 \
  --include_experimental \
  --disable_arrival_simulation \
  --output_dir results/smoke_fake
```

This verifies the benchmark pipeline without downloading models or datasets.

## HF Benchmark Example

CPU-friendly example:

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model distilgpt2 \
  --device cpu \
  --workload public_dataset \
  --dataset ag_news \
  --prompt_mode templated \
  --n_requests 32 \
  --include_experimental \
  --disable_arrival_simulation \
  --output_dir results/hf_cpu_agnews_templated
```

GPU example:

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda:0 \
  --dtype float16 \
  --workload public_dataset \
  --dataset ag_news \
  --prompt_mode templated \
  --n_requests 64 \
  --include_experimental \
  --disable_arrival_simulation \
  --output_dir results/hf_qwen_agnews_templated
```

If using older GPUs such as NVIDIA P100, use a PyTorch wheel that supports compute capability `sm_60`. Modern vLLM/SGLang wheels may require newer GPUs.

## Reproduce A Small Matrix

This reproduces one model/dataset across the three prompt modes and three seeds:

```bash
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="ag_news"
N=64
OUT="results/reproduction_qwen_agnews"

for SEED in 42 123 456; do
  for MODE in raw templated semantic; do
    python experiments/run_benchmark.py \
      --backend hf \
      --model "$MODEL" \
      --device cuda:0 \
      --dtype float16 \
      --workload public_dataset \
      --dataset "$DATASET" \
      --prompt_mode "$MODE" \
      --n_requests "$N" \
      --seed "$SEED" \
      --include_experimental \
      --disable_arrival_simulation \
      --output_dir "$OUT/$MODE/seed_$SEED"
  done
done
```

Then summarize:

```bash
python experiments/analyze_shadowkv_results.py "$OUT"
```

## Reproduce The Full Harness Evaluation

The full paper-style sweep is expensive because it spans five models, ten datasets, three prompt modes, and three seeds on GPU. Use this only on a machine with sufficient GPU availability and model access:

```bash
MODELS=(
  "gpt2"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "google/gemma-2b-it"
  "microsoft/Phi-3-mini-4k-instruct"
)

DATASETS=(
  ag_news banking77 alpaca_eval dolly daily_dialog
  oasst1 ultrachat samsum xsum cnn_dailymail
)

MODES=(raw templated semantic)
SEEDS=(42 123 456)
N=64
OUT="results/full_reproduction"

for MODEL in "${MODELS[@]}"; do
  MODEL_DIR=$(echo "$MODEL" | tr '/.' '__')
  for MODE in "${MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for DATASET in "${DATASETS[@]}"; do
        python experiments/run_benchmark.py \
          --backend hf \
          --model "$MODEL" \
          --device cuda:0 \
          --dtype float16 \
          --workload public_dataset \
          --dataset "$DATASET" \
          --prompt_mode "$MODE" \
          --n_requests "$N" \
          --seed "$SEED" \
          --include_experimental \
          --disable_arrival_simulation \
          --output_dir "$OUT/$MODEL_DIR/$MODE/seed_$SEED/$DATASET"
      done
    done
  done
done
```

The exact latency values will vary by GPU, driver, PyTorch, Transformers version, and Hugging Face dataset/model cache state.

## Important Correctness Boundary

Exact-prefix KV reuse is semantics-preserving under causal attention. Approximate semantic KV reuse is not generally correctness-preserving.

ShadowKV++ therefore separates:

- semantic opportunity detection,
- utility scoring,
- execution admission,
- and backend correctness validation.

For real HF backends, report semantic opportunity metrics separately from exact-prefix cache-hit metrics unless a backend-specific guard validates approximate reuse.

## Citation

If you use this artifact, cite the associated ShadowKV++ manuscript and include the result manifest:

```text
results/manifest.json
```

## Acknowledgements

The included P100 experiments were run using NVIDIA P100 GPU access provided by Prof. Sparsh Mittal and the Department of Electronics and Communication Engineering, IIT Roorkee.
