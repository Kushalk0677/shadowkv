#!/usr/bin/env python3
"""
Direction 1: Semantic Fidelity — Proper Equivalence-Key Method.

Uses the paper's SEMANTIC_PARAPHRASE_TEMPLATES to compare outputs between
semantically equivalent but token-different instruction variants.
This is the correct methodology for measuring semantic KV reuse fidelity.
"""

import argparse, json, os, sys, time, warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Models (same as the paper's HF evaluation)
MODELS = {
    'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'qwen25_15b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'gemma2b': 'google/gemma-2b-it',
}

# Dataset → HF name → split → field → dataset_type (for template lookup)
# Format: (name, split, field, dataset_type) for parquet-based
#         (name, subname, split, field, dataset_type) for script-based
DATASETS = {
    'samsum': ('knkarthick/samsum', 'train', 'dialogue', 'dialogue_summary'),
    'xsum': ('xsum', 'xsum', 'train', 'document', 'summarization'),
    'ag_news': ('fancyzhx/ag_news', 'train', 'text', 'classification'),
    'banking77': ('mteb/banking77', 'train', 'text', 'classification'),
    'alpaca_eval': ('Thanmay/alpaca_eval', 'eval', 'instruction', 'instruction'),
    'dolly': ('databricks/databricks-dolly-15k', 'train', 'instruction', 'instruction'),
    'daily_dialog': ('DeepPavlov/daily_dialog', 'train', 'dialog', 'dialogue_summary'),
    'oasst1': ('OpenAssistant/oasst1', 'train', 'text', 'oasst'),
    'ultrachat': ('HuggingFaceH4/ultrachat_200k', 'train_sft', 'messages', 'chat_messages'),
}

# The paper's semantic paraphrase templates (4 variants each)
SEMANTIC_PARAPHRASE_TEMPLATES = {
    'dialogue_summary': (
        'System: Condense the following conversation into a factual one or two sentence recap. Preserve speaker intent and outcome.\nConversation payload:\n',
        'Instruction: Read the dialogue and write a compact neutral summary. Keep only the goal, action, and resolution.\nDialogue input:\n',
        'Role: Dialogue summarizer. Ignore greetings and filler, then summarize the exchange faithfully in brief.\nTranscript:\n',
        'Task brief: Create a concise summary of this multi-turn dialogue without adding facts.\nRequest body:\n',
    ),
    'instruction': (
        'System: Follow the user instruction and provide a concise helpful answer based only on supplied context.\nRequest:\n',
        'Instruction-following task: identify what is being asked and answer directly, avoiding invented details.\nPayload:\n',
        'Assistant policy: be useful, brief, and faithful to the instruction and context below.\nInput:\n',
        'Task: produce a short operationally useful response to the following instruction.\nUser material:\n',
    ),
    'classification': (
        'System: Classify the item by its dominant intent or topic and output the most likely label.\nItem payload:\n',
        'Classification task: identify the category that best matches the following text.\nText:\n',
        'Decision brief: read the item, ignore incidental wording, and emit one category.\nInput item:\n',
        'Task: choose the strongest topic or intent label for the text below.\nRequest body:\n',
    ),
    'oasst': (
        'System: Continue the conversation helpfully and safely while preserving the user intent and language.\nConversation item:\n',
        'Assistant continuation task: respond naturally, safely, and concisely to the message below.\nMessage payload:\n',
        'Safety-aware chat brief: infer the next helpful assistant turn from the supplied text.\nConversation payload:\n',
        'Task: produce a safe compact assistant reply that follows the conversation context.\nInput:\n',
    ),
    'chat_messages': (
        'System: Read the chat transcript and produce the next helpful assistant turn.\nTranscript payload:\n',
        'Chat continuation instruction: preserve topic continuity and answer the last user need.\nConversation:\n',
        'Assistant role: continue this dialogue naturally, grounded only in the provided messages.\nMessages:\n',
        'Task brief: infer and write the next concise helpful chat response.\nDialogue:\n',
    ),
    'summarization': (
        'System: Summarize the document in one factual sentence focused on the main event.\nDocument payload:\n',
        'News summary task: extract the central event and produce a compact factual summary.\nArticle:\n',
        'Editorial brief: preserve core facts, avoid speculation, and write a short summary.\nSource text:\n',
        'Task: read the article and return a concise factual synopsis.\nInput document:\n',
    ),
    'generic': (
        'System: Answer the following request concisely and faithfully.\nPayload:\n',
        'Assistant task: read the request and respond directly.\nInput:\n',
        'Instruction: provide a brief grounded answer to the material below.\nRequest:\n',
        'Task brief: produce a useful concise response.\nBody:\n',
    ),
}

MAX_GEN_TOKENS = 64
NUM_SAMPLES = 32

# Map dataset key to dataset_type for template lookup
DATASET_TYPE = {}
for k, v in DATASETS.items():
    DATASET_TYPE[k] = v[-1]


def load_model(model_name: str, device: str):
    print(f'Loading {model_name} on {device}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
        low_cpu_mem_usage=True
    ).to(device).eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, device: str, max_new: int = MAX_GEN_TOKENS) -> str:
    """Generate text from a prompt using greedy decoding."""
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def get_field_idx(ds_key: str) -> int:
    """Get the field index from the DATASETS tuple (3 for 5-tuples, 2 for 4-tuples)."""
    return 3 if len(DATASETS[ds_key]) >= 5 else 2


def extract_payload(row, ds_key: str) -> str:
    """Extract the payload text from a dataset row, handling different formats."""
    if ds_key == 'daily_dialog':
        dialog = row.get('dialog', [])
        return ' '.join(dialog) if isinstance(dialog, list) else str(dialog)
    elif ds_key == 'ultrachat':
        msgs = row.get('messages', [])
        texts = [m.get('content', '') for m in msgs if isinstance(m, dict)]
        return ' '.join(texts)
    elif ds_key in ('alpaca_eval', 'dolly'):
        return str(row.get('instruction', ''))
    elif ds_key == 'oasst1':
        return str(row.get('text', ''))
    else:
        return str(row[DATASETS[ds_key][get_field_idx(ds_key)]])


def run_fidelity_experiment(args):
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_key in args.models:
        model_name = MODELS[model_key]
        model, tokenizer = load_model(model_name, device)

        for ds_key in args.datasets:
            ds_info = DATASETS[ds_key]
            ds_name = ds_info[0]
            ds_type = DATASET_TYPE[ds_key]
            templates = SEMANTIC_PARAPHRASE_TEMPLATES.get(ds_type, SEMANTIC_PARAPHRASE_TEMPLATES['generic'])
            n_variants = len(templates)

            print(f'\n=== {model_key} on {ds_key} ({ds_type}, {n_variants} variants) ===')

            # Load dataset with error handling for all formats
            samples = []
            try:
                from datasets import load_dataset
                if ds_key in ('xsum',):
                    from huggingface_hub import hf_hub_download
                    import pyarrow.parquet as pq
                    pp = hf_hub_download(repo_id='xsum', filename='data/train-00000-of-00001.parquet', repo_type='dataset')
                    table = pq.read_table(pp)
                    for i in range(min(NUM_SAMPLES, len(table))):
                        text = str(table.column('document')[i].as_py() or '')
                        if len(text) > 20:
                            samples.append(text[:512])
                else:
                    # Try loading; if trust_remote_code warning appears it's harmless for parquet datasets
                    ds_args = [ds_name]
                    split_name = ds_info[2] if len(ds_info) >= 5 else ds_info[1]
                    if len(ds_info) >= 5:
                        ds_args.append(ds_info[1])
                    ds = load_dataset(*ds_args, split=split_name, trust_remote_code=True)
                    for i in range(min(NUM_SAMPLES, len(ds))):
                        payload = extract_payload(ds[i], ds_key)
                        if len(payload) > 20:
                            samples.append(payload[:512])
            except Exception as e:
                print(f'  Could not load dataset: {e}')
                continue

            print(f'  Loaded {len(samples)} samples')
            if not samples:
                continue

            # For each sample, generate with variant A (exact) and variant B (semantic)
            for idx, payload in enumerate(samples):
                variant_a = 0  # first variant = exact prefix
                variant_b = (idx % (n_variants - 1)) + 1  # cycle through variants 1,2,3

                prompt_a = templates[variant_a] + payload
                prompt_b = templates[variant_b] + payload

                try:
                    text_a = generate_text(model, tokenizer, prompt_a, device, args.max_gen_tokens)
                    time.sleep(0.01)  # tiny cooldown
                    text_b = generate_text(model, tokenizer, prompt_b, device, args.max_gen_tokens)
                except Exception as e:
                    print(f'  Error on sample {idx}: {e}')
                    continue

                all_results.append({
                    'model': model_key,
                    'dataset': ds_key,
                    'sample_idx': idx,
                    'variant_a': variant_a,
                    'variant_b': variant_b,
                    'exact_text': text_a,
                    'approx_text': text_b,
                    'exact_len': len(text_a.split()),
                    'approx_len': len(text_b.split()),
                })

                if (idx + 1) % 10 == 0:
                    print(f'  Processed {idx+1}/{len(samples)}')

        # Save per-model results
        model_out = output_dir / f'{model_key}_equiv_results.json'
        model_results = [r for r in all_results if r['model'] == model_key]
        with open(model_out, 'w') as f:
            json.dump(model_results, f, indent=2)
        print(f'Saved {len(model_results)} results to {model_out}')

        del model, tokenizer
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Save all results
    all_out = output_dir / 'all_results.json'
    with open(all_out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nTotal: {len(all_results)} results saved to {all_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Fidelity — Equivalence-Key Method')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--models', nargs='+', default=['tinyllama'], choices=MODELS.keys())
    parser.add_argument('--datasets', nargs='+', default=['samsum', 'alpaca_eval', 'banking77'],
                       choices=DATASETS.keys())
    parser.add_argument('--n_samples', type=int, default=32)
    parser.add_argument('--max_gen_tokens', type=int, default=64)
    parser.add_argument('--output_dir', default='fidelity_equiv_results')
    args = parser.parse_args()
    run_fidelity_experiment(args)
