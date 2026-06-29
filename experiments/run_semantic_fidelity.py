#!/usr/bin/env python3
"""
Direction 1: Semantic Fidelity Measurement Pipeline

Measures the output fidelity of approximate semantic KV reuse by comparing
generated text from exact-prefix vs semantic-approximate KV caches.

Usage on P100:
  python run_semantic_fidelity.py --device cuda:0 --models gpt2,qwen25_15b --output_dir fidelity_results

Dependencies: torch, transformers, datasets, bert-score, rouge-score
"""

import argparse, json, os, sys, time, random
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Models that fit on P100 (12GB)
P100_MODELS = {
    'gpt2': 'gpt2',
    'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'qwen25_15b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'gemma2b': 'google/gemma-2b-it',
    'phi3mini': 'microsoft/Phi-3-mini-4k-instruct',
}

DATASETS = {
    'samsum': ('knkarthick/samsum', 'train', 'dialogue'),
    'xsum': ('xsum', 'xsum', 'train', 'document'),
    'cnn_dailymail': ('cnn_dailymail', 'cnn_dailymail', 'train', 'article'),
    'ag_news': ('ag_news', 'ag_news', 'train', 'text'),
    'banking77': ('mteb/banking77', 'train', 'text'),
    'alpaca_eval': ('Thanmay/alpaca_eval', 'eval', 'instruction'),
    'dolly': ('databricks/databricks-dolly-15k', 'train', 'instruction'),
    'daily_dialog': ('DeepPavlov/daily_dialog', 'train', 'dialog'),
    'oasst1': ('OpenAssistant/oasst1', 'train', 'text'),
    'ultrachat': ('HuggingFaceH4/ultrachat_200k', 'train_sft', 'messages'),
}

MAX_GEN_TOKENS = 64
NUM_SAMPLES = 50


def load_model(model_name: str, device: str):
    print(f'Loading {model_name} on {device}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32
    ).to(device).eval()
    return model, tokenizer


def generate_text(model, tokenizer, input_ids: torch.Tensor, device: str, max_new: int = MAX_GEN_TOKENS) -> str:
    """Generate text from a prompt, return decoded string."""
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def prefill_and_generate(model, tokenizer, prompt: str, device: str,
                          past_key_values=None, max_new: int = MAX_GEN_TOKENS) -> Tuple[str, object]:
    """Prefill prompt, optionally with past_key_values, then generate."""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    if past_key_values is not None:
        # Incremental prefill with KV cache
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], past_key_values=past_key_values, use_cache=True)
        past = outputs.past_key_values
        input_for_gen = torch.tensor([[]], dtype=torch.long, device=device)
    else:
        # Full prefill without cache
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], use_cache=True)
        past = outputs.past_key_values
        input_for_gen = inputs['input_ids']

    # Generate
    generated_ids = []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(input_ids=input_for_gen if len(generated_ids) == 0 else last_token,
                        past_key_values=past if len(generated_ids) > 0 else None,
                        use_cache=True)
        logits = out.logits[0, -1]
        next_token = logits.argmax(-1).unsqueeze(0).unsqueeze(0)
        generated_ids.append(next_token.item())
        last_token = next_token
        past = out.past_key_values
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True), past


def run_fidelity_experiment(args):
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model_key in args.models:
        model_name = P100_MODELS[model_key]
        model, tokenizer = load_model(model_name, device)

        for ds_key in args.datasets:
            ds_info = DATASETS[ds_key]
            print(f'\n=== {model_key} on {ds_key} ===')

            try:
                from datasets import load_dataset
                
                # Script-based datasets need special handling across datasets versions
                script_based = ds_key in ('xsum', 'cnn_dailymail', 'ag_news')
                
                if script_based:
                    # Load via huggingface_hub directly (works on ALL datasets versions)
                    from huggingface_hub import hf_hub_download
                    import pandas as pd
                    import pyarrow.parquet as pq
                    
                    hub_paths = {
                        'xsum': ('xsum', 'data/train-00000-of-00001.parquet', 'document'),
                        'cnn_dailymail': ('cnn_dailymail', 'data/train-00000-of-00001.parquet', 'article'),
                        'ag_news': ('fancyzhx/ag_news', 'data/train-00000-of-00001.parquet', 'text'),
                    }
                    repo_id, parquet_file, field = hub_paths[ds_key]
                    parquet_path = hf_hub_download(repo_id=repo_id, filename=parquet_file, repo_type='dataset')
                    table = pq.read_table(parquet_path)
                    samples = []
                    for i in range(min(NUM_SAMPLES, len(table))):
                        text = str(table.column(field)[i].as_py() or '')
                        if len(text) > 20:
                            samples.append(text[:512])
                    print(f'  Loaded {len(samples)} samples via HF Hub')
                    if not samples:
                        raise ValueError('No samples loaded from parquet')
                else:
                    # Parquet-based datasets work normally
                    ds_name = ds_info[0]
                    if len(ds_info) == 4:
                        ds = load_dataset(ds_name, ds_info[1], split=ds_info[2], trust_remote_code=True)
                        field_idx = 3
                    else:
                        ds = load_dataset(ds_name, split=ds_info[1], trust_remote_code=True)
                        field_idx = 2
                    # Get samples — handle different dataset formats
                    samples = []
                    for i in range(min(NUM_SAMPLES, len(ds))):
                        row = ds[i]
                        if ds_key == 'daily_dialog':
                            text = ' '.join(row.get('dialog', [])) if isinstance(row.get('dialog'), list) else str(row.get('dialog', ''))
                        elif ds_key == 'ultrachat':
                            msgs = row.get('messages', [])
                            text = ' '.join(m.get('content', '') for m in msgs if isinstance(m, dict)) if isinstance(msgs, list) else ''
                        elif ds_key == 'oasst1':
                            text = str(row.get('text', ''))
                        elif ds_key in ('alpaca_eval', 'dolly'):
                            text = str(row.get('instruction', ''))
                        else:
                            text = str(row[ds_info[field_idx]])
                        if text and len(text) > 20:
                            samples.append(text[:512])  # truncate to 512 chars
            except Exception as e:
                print(f'  Could not load dataset {ds_key}: {e}')
                continue

            print(f'  Loaded {len(samples)} samples')

            for idx, prompt_text in enumerate(samples[:args.n_samples]):
                # Truncate to reasonable length
                tokens = tokenizer.encode(prompt_text, truncation=True, max_length=256)
                prompt = tokenizer.decode(tokens[:128])

                # --- Exact prefill (baseline) ---
                try:
                    exact_text, exact_past = prefill_and_generate(
                        model, tokenizer, prompt, device, max_new=args.max_gen_tokens
                    )
                except Exception as e:
                    print(f'  Error in exact prefill: {e}')
                    continue

                # --- Semantic-approximate prefill ---
                # Use a prefix-shuffled version of the prompt to simulate semantic similarity
                # This creates a token-different but semantically similar prefix
                approx_prompt = prompt
                if args.approximation == 'prefix_shuffle':
                    # Shuffle the last 25% of tokens to create a semantic-approximate prefix
                    t = tokenizer.encode(prompt)
                    split = len(t) * 3 // 4
                    prefix = t[:split]
                    suffix = t[split:]
                    random.shuffle(suffix)
                    approx_prompt = tokenizer.decode(prefix + suffix)
                elif args.approximation == 'prefix_truncate':
                    # Use a shorter prefix (simulates partial match)
                    t = tokenizer.encode(prompt)
                    shorter = t[:len(t) * 3 // 4]
                    approx_prompt = tokenizer.decode(shorter)

                try:
                    approx_text, _ = prefill_and_generate(
                        model, tokenizer, approx_prompt, device, max_new=args.max_gen_tokens
                    )
                except Exception as e:
                    print(f'  Error in approx prefill: {e}')
                    continue

                results.append({
                    'model': model_key,
                    'dataset': ds_key,
                    'sample_idx': idx,
                    'prompt': prompt[:200],
                    'exact_text': exact_text,
                    'approx_text': approx_text,
                    'exact_len': len(exact_text.split()),
                    'approx_len': len(approx_text.split()),
                })

                if (idx + 1) % 10 == 0:
                    print(f'  Processed {idx+1}/{len(samples[:args.n_samples])}')

        # Save per-model results
        model_out = output_dir / f'{model_key}_results.json'
        model_results = [r for r in results if r['model'] == model_key]
        with open(model_out, 'w') as f:
            json.dump(model_results, f, indent=2)
        print(f'Saved {len(model_results)} results to {model_out}')

        # Clean up GPU memory
        del model
        del tokenizer
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Save all results
    all_out = output_dir / 'all_results.json'
    with open(all_out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nTotal results: {len(results)} saved to {all_out}')
    print('Run evaluation: python eval_fidelity.py --input ' + str(all_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Fidelity Measurement')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--models', nargs='+', default=['gpt2', 'qwen25_15b'], choices=P100_MODELS.keys())
    parser.add_argument('--datasets', nargs='+', default=['samsum', 'xsum'], choices=DATASETS.keys())
    parser.add_argument('--n_samples', type=int, default=30)
    parser.add_argument('--max_gen_tokens', type=int, default=64)
    parser.add_argument('--approximation', choices=['prefix_shuffle', 'prefix_truncate'], default='prefix_shuffle')
    parser.add_argument('--output_dir', default='fidelity_results')
    args = parser.parse_args()
    run_fidelity_experiment(args)
