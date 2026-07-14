#!/usr/bin/env python3
"""
Semantic Fidelity — KV Cache Reuse using DynamicCache API.

All 5 models, all 10 datasets. Measures whether splicing cached KV from
one prompt prefix into another changes the generated output.
"""
import argparse, json, os, sys, time, random, warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    'gpt2': 'gpt2',
    'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'qwen25_15b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'gemma2b': 'google/gemma-2b-it',
    'phi3mini': 'microsoft/Phi-3-mini-4k-instruct',
}

# All 10 datasets from the paper's evaluation
DATASETS = {
    'samsum': ('knkarthick/samsum', 'train', 'dialogue'),
    'xsum': ('xsum', 'train', 'document'),           # script-based
    'cnn_dailymail': ('abisee/cnn_dailymail', '3.0.0', 'train', 'article'),
    'ag_news': ('fancyzhx/ag_news', 'train', 'text'),
    'banking77': ('mteb/banking77', 'train', 'text'),
    'alpaca_eval': ('Thanmay/alpaca_eval', 'eval', 'instruction'),
    'dolly': ('databricks/databricks-dolly-15k', 'train', 'instruction'),
    'daily_dialog': ('DeepPavlov/daily_dialog', 'train', 'dialog'),
    'oasst1': ('OpenAssistant/oasst1', 'train', 'text'),
    'ultrachat': ('HuggingFaceH4/ultrachat_200k', 'train_sft', 'messages'),
}

# Datasets that need parquet-based loading (script-based on HF)
PARQUET_DATASETS = {
    'xsum': ('xsum', 'data/train-00000-of-00001.parquet', 'document'),
}


def load_model(model_name, device):
    print(f'Loading {model_name} on {device}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
        low_cpu_mem_usage=True
    ).to(device).eval()
    return model, tokenizer


def generate_standard(model, input_ids, max_new):
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, max_new_tokens=max_new,
            do_sample=False, num_beams=1,
            pad_token_id=model.config.pad_token_id or 0,
        )
    return out[0][input_ids.shape[1]:]


def generate_with_reuse(model, prompt_ids_orig, prompt_ids_mod, shared, max_new, eos_id):
    shared = min(shared, len(prompt_ids_orig[0]), len(prompt_ids_mod[0]))

    with torch.no_grad():
        out = model(prompt_ids_orig, use_cache=True)
    cache = out.past_key_values
    cache.crop(shared)

    suffix = prompt_ids_mod[:, shared:]
    if suffix.shape[1] > 0:
        with torch.no_grad():
            out = model(suffix, past_key_values=cache, use_cache=True)
        combined_cache = out.past_key_values
        total_pos = shared + suffix.shape[1]
        start_tok = suffix[:, -1:]
    else:
        combined_cache = cache
        total_pos = shared
        start_tok = prompt_ids_orig[:, shared - 1:shared]

    if total_pos < 1:
        return generate_standard(model, prompt_ids_mod, max_new)

    kv = combined_cache
    kv.crop(total_pos - 1)
    inp = start_tok
    gen_ids = []
    for _ in range(max_new):
        with torch.no_grad():
            outs = model(input_ids=inp, past_key_values=kv, use_cache=True)
        nid = outs.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
        inp = nid
        kv = outs.past_key_values
        gen_ids.append(nid.item())
        if nid.item() == eos_id:
            break
    return torch.tensor(gen_ids)


def make_shuffled_ids(ids, ratio=0.75):
    split = int(len(ids) * ratio)
    if len(ids) - split < 4:
        split = len(ids) - 4
    if split < 4:
        return ids, len(ids)
    prefix = ids[:split]
    suffix = ids[split:]
    shuffled = suffix.copy()
    random.shuffle(shuffled)
    return prefix + shuffled, split


def extract(row, key):
    """Extract payload text from a dataset row, handling different formats."""
    if key in ('alpaca_eval', 'dolly'):
        return str(row.get('instruction', ''))
    elif key == 'samsum':
        return str(row.get('dialogue', ''))
    elif key == 'xsum':
        return str(row.get('document', ''))
    elif key == 'cnn_dailymail':
        return str(row.get('article', ''))
    elif key in ('ag_news', 'banking77'):
        return str(row.get('text', ''))
    elif key == 'daily_dialog':
        dialog = row.get('dialog', [])
        return ' '.join(dialog) if isinstance(dialog, list) else str(dialog)
    elif key == 'oasst1':
        return str(row.get('text', ''))
    elif key == 'ultrachat':
        msgs = row.get('messages', [])
        texts = [m.get('content', '') for m in msgs if isinstance(m, dict)]
        return ' '.join(texts)
    return str(row.get(list(row.keys())[0], ''))


def load_dataset_samples(dk, n_samples):
    """Load samples from a dataset, handling script-based datasets.
    Loads extra rows to ensure n_samples valid samples are returned."""
    if dk in PARQUET_DATASETS:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        repo_id, parquet_file, field = PARQUET_DATASETS[dk]
        pp = hf_hub_download(repo_id=repo_id, filename=parquet_file, repo_type='dataset')
        table = pq.read_table(pp)
        samples = []
        for i in range(min(n_samples * 3, len(table))):
            text = str(table.column(field)[i].as_py() or '')
            if len(text) > 5:
                samples.append(text[:512])
                if len(samples) >= n_samples:
                    break
        return samples
    else:
        from datasets import load_dataset
        info = DATASETS[dk]
        ds_name = info[0]
        if len(info) == 4:
            ds = load_dataset(ds_name, info[1], split=info[2], trust_remote_code=True)
        else:
            ds = load_dataset(ds_name, split=info[1], trust_remote_code=True)
        samples = []
        for i in range(min(n_samples * 3, len(ds))):
            t = extract(ds[i], dk)
            if len(t) > 5:
                samples.append(t[:512])
                if len(samples) >= n_samples:
                    break
        return samples


def run_single_ratio(mk, model, tok, dk, samples, ratio, max_gen_tokens, device):
    eos_id = tok.eos_token_id or 0
    results = []
    for idx, payload in enumerate(samples):
        ids_1d = tok.encode(payload, truncation=True, max_length=384)
        if len(ids_1d) < 16:
            continue
        ids_mod_1d, shared = make_shuffled_ids(ids_1d, ratio=ratio)
        ids_orig = torch.tensor([ids_1d], device=device)
        ids_mod = torch.tensor([ids_mod_1d], device=device)

        exact_ids = generate_standard(model, ids_orig, max_gen_tokens)
        exact_text = tok.decode(exact_ids, skip_special_tokens=True)

        ref_ids = generate_standard(model, ids_mod, max_gen_tokens)
        ref_text = tok.decode(ref_ids, skip_special_tokens=True)

        reuse_ids = generate_with_reuse(model, ids_orig, ids_mod, shared,
                                         max_gen_tokens, eos_id)
        reuse_text = tok.decode(reuse_ids, skip_special_tokens=True)

        results.append({
            'model': mk, 'dataset': dk, 'sample_idx': idx,
            'shared_ratio': ratio,
            'shared_tokens': shared,
            'total_tokens_orig': len(ids_1d),
            'total_tokens_mod': len(ids_mod_1d),
            'exact_text': exact_text,
            'ref_text': ref_text,
            'reuse_text': reuse_text,
        })
        if (idx + 1) % 5 == 0:
            print(f'  [{idx+1}/{len(samples)}] ratio={ratio}')
    return results


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    random.seed(42)
    device = args.device

    for mk in args.models:
        model, tok = load_model(MODELS[mk], device)

        for dk in args.datasets:
            print(f'\n=== {mk} on {dk} ===')
            samples = load_dataset_samples(dk, args.n_samples)
            print(f'  {len(samples)} samples')
            if not samples:
                continue

            for idx, payload in enumerate(samples):
                ids_1d = tok.encode(payload, truncation=True, max_length=384)
                if len(ids_1d) < 16:
                    continue
                for ratio in args.ratios:
                    chunk = run_single_ratio(mk, model, tok, dk, [payload],
                                             ratio, args.max_gen_tokens, device)
                    all_results.extend(chunk)

        path = out_dir / f'{mk}_results.json'
        with open(path, 'w') as f:
            json.dump([r for r in all_results if r['model'] == mk], f, indent=2)
        print(f'Saved -> {path}')
        del model, tok
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    path = out_dir / 'all_results.json'
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Total: {len(all_results)} -> {path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--models', nargs='+', default=['tinyllama'], choices=MODELS.keys())
    p.add_argument('--datasets', nargs='+', default=['samsum', 'alpaca_eval', 'banking77'],
                   choices=DATASETS.keys())
    p.add_argument('--n_samples', type=int, default=128)
    p.add_argument('--max_gen_tokens', type=int, default=64)
    p.add_argument('--ratios', nargs='+', type=float, default=[0.75])
    p.add_argument('--output_dir', default='fidelity_equiv_results')
    args = p.parse_args()
    run(args)
