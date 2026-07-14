# Experimental Setup: Semantic Fidelity Measurement

## 1. Research Question

Does splicing cached key-value (KV) cache from one prompt prefix into another prompt's computation change the generated output? This measures the output fidelity of the MeritKV `_partial_semantic_reuse` mechanism.

## 2. Methodology

### 2.1 Core Experiment

We use `DynamicCache.crop()` (transformers ≥ 5.0) to perform zero-copy KV cache splicing:

```
1. Prefill prompt A → capture DynamicCache (seq_len = N)
2. cache.crop(shared) → keep first S tokens (shared prefix)
3. Prefill prompt B's suffix (tokens S..N-1) on cropped cache
4. Generate tokens from combined state
5. Compare output to clean generation from prompt B
```

Three texts are collected per sample:

| Label | Method | Purpose |
|-------|--------|---------|
| `exact_text` | Generate from prompt A (original) | Upper bound reference |
| `ref_text` | Generate from prompt B (clean, no reuse) | Fair baseline |
| `reuse_text` | Generate from prompt B **using A's KV cache** | KV reuse test |

**Primary metric**: `ref_text vs reuse_text` — does KV reuse change the output?

**Secondary metric**: `exact_text vs ref_text` — how much does prompt rephrasing alone change output? (Prompt sensitivity baseline)

### 2.2 Prefix Shuffle

Prompt B is created by shuffling the last 25% of prompt A's tokens:

```python
split = len(tokens) * 3 // 4
prefix = tokens[:split]       # Identical to A (shared prefix)
suffix = tokens[split:]        # Shuffled (simulates semantic mismatch)
random.shuffle(suffix)
tokens_B = prefix + suffix
```

This preserves a token-identical prefix (75%) while making the suffix semantically different — modeling what happens when two similar requests share a common scaffold.

### 2.3 Control Experiment (ratio = 0.0)

When `shared_ratio = 0.0`, ALL tokens are shuffled and no cache is reused:

```
cache.crop(0)    → empty cache
prefill(B, past_key_values=None)  → same as clean generation
```

This should always produce `ref_text == reuse_text` (ROUGE-L = 1.0), confirming the pipeline is free of implementation bugs.

### 2.4 Implementation Details

```python
def generate_with_reuse(model, prompt_A, prompt_B, shared):
    # 1. Prefill original prompt
    out = model(prompt_A, use_cache=True)
    cache = out.past_key_values

    # 2. Crop to shared prefix
    cache.crop(shared)

    # 3. Prefill modified suffix on cropped cache
    suffix = prompt_B[:, shared:]
    if suffix.shape[1] > 0:
        out = model(suffix, past_key_values=cache, use_cache=True)
        combined_cache = out.past_key_values
        total_pos = shared + suffix.shape[1]
        start_tok = suffix[:, -1:]
    else:
        combined_cache = cache
        total_pos = shared

    # 4. Prepare for generation: cache covers positions 0..total_pos-2
    combined_cache.crop(total_pos - 1)

    # 5. Generate token-by-token
    for _ in range(max_new):
        out = model(input_ids=start_tok, past_key_values=combined_cache, use_cache=True)
        next_token = out.logits[:, -1].argmax(-1)
        combined_cache = out.past_key_values
        start_tok = next_token
```

Key subtleties:
- Token arrays are manipulated directly (no `decode/encode` roundtrip) to preserve exact token alignment.
- `A` manual generation loop is used instead of `model.generate()` because transformers 5.x has compatibility issues with single-token `input_ids + past_key_values`.
- The cache is cropped to `total_pos - 1` before generation because the last prompt token is passed as `input_ids` — the cache should only contain preceding tokens.

## 3. Models

Five models spanning three architecture families:

| Model | Params | Architecture | Source |
|-------|--------|-------------|--------|
| GPT-2 | 124M | GPT-2 (decoder) | OpenAI |
| TinyLlama | 1.1B | LLaMA | TinyLlama project |
| Qwen 2.5 1.5B | 1.5B | Qwen2 | Alibaba |
| Gemma 2B | 2.0B | Gemma | Google DeepMind |
| Phi-3 Mini | 3.8B | Phi-3 (decoder) | Microsoft |

Each model is loaded in its deployment precision: float32 for CPU experiments, float16 for GPU (T4) experiments.

## 4. Datasets

Ten datasets covering four task types:

| Dataset | Task Type | Source | Target Samples |
|---------|-----------|--------|----------------|
| samsum | Dialogue summarization | Samsung | 128 |
| xsum | News summarization | BBC | 128 |
| cnn_dailymail | News summarization | CNN/DailyMail | 128 |
| ag_news | Text classification | AG's Corpus | 128 |
| banking77 | Intent classification | Banking | 128 |
| alpaca_eval | Instruction following | Stanford | 128 |
| dolly | Instruction following | Databricks | 128 |
| daily_dialog | Dialogue generation | DeepPavlov | 128 |
| oasst1 | Conversation | OpenAssistant | 128 |
| ultrachat | Chat | HuggingFaceH4 | 128 |

Each dataset is loaded via HuggingFace `datasets`. The loading loop scans up to 3× the target sample count to ensure `n` valid samples are collected. The minimum text length threshold is 5 characters (reduced from 20 in preliminary runs to maximize sample recovery for short-text datasets like banking77).

## 5. Metrics

### 5.1 Exact String Match
```
match = (ref_text == reuse_text)
```
Strictest metric — any token difference counts as a failure. Binary (per-sample).

### 5.2 ROUGE-L F1
```
ROUGE-L = rouge_scorer.score(ref_text, reuse_text)['rougeL'].fmeasure
```
Token-sequence overlap with longest common subsequence. Measures content similarity, tolerates minor format differences.

Both metrics are computed on the raw generated text (after removing special tokens).

## 6. Precision Comparison

Experiments are run in two precision modes:

| Mode | Hardware | Precision | Purpose |
|------|----------|-----------|---------|
| float32 | CPU | 32-bit floating point | Upper bound on numerical accuracy |
| float16 | GPU (T4) | 16-bit floating point | Deployment-realistic precision |

Comparing both reveals whether fidelity measurements are precision-dependent.

## 7. Reproducibility

- Random seed: 42
- Decoding: greedy (do_sample=False, num_beams=1)
- Max generation tokens: 64
- Max input tokens: 384
- All experiments use the same seed and parameters
