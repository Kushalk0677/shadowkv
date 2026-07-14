# Semantic Fidelity Measurement — Direction 1

Measures output quality of approximate semantic KV reuse by comparing
generated text from exact-prefix vs semantic-approximate KV caches.

## Setup (on P100 server)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r fidelity_requirements.txt
```

## Run the experiment

```bash
python run_semantic_fidelity.py \
  --device cuda:0 \
  --models gpt2 qwen25_15b \
  --datasets samsum xsum \
  --n_samples 30 \
  --max_gen_tokens 64 \
  --output_dir fidelity_results
```

Models that fit on P100 (12 GB): gpt2, tinyllama, qwen25_15b, gemma2b, phi3mini

## Evaluate

```bash
pip install bert-score rouge-score
python eval_fidelity.py \
  --input fidelity_results/all_results.json \
  --output fidelity_results/summary.json
```

## Expected output

- `fidelity_results/all_results.json` — per-sample exact vs approx text with ROUGE-L
- `fidelity_results/summary.json` — aggregate ROUGE-L F1, exact match rate, BERTScore F1
- A high ROUGE-L F1 (>0.85) with low speedup drop indicates semantic reuse preserves quality

## Notes

- Uses greedy decoding (deterministic) so differences are from KV cache, not sampling
- `prefix_shuffle` approximation shuffles the last 25% of prompt tokens
- `prefix_truncate` truncates 25% of the prefix
- Both simulate real-world semantic-approximate matches
