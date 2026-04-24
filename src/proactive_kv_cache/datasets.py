from __future__ import annotations

from typing import Dict, List



DATASET_REGISTRY = {
    'daily_dialog': {
        'hf_name': 'DeepPavlov/daily_dialog',
        'default_split': 'train',
        'type': 'dialogue_summary',
    },
    'samsum': {
        'hf_name': 'knkarthick/samsum',
        'default_split': 'train',
        'type': 'dialogue_summary',
    },
    'dolly': {
        'hf_name': 'databricks/databricks-dolly-15k',
        'default_split': 'train',
        'type': 'instruction',
    },
    'alpaca_eval': {
        # This mirror loads without the old dataset script.
        'hf_name': 'Thanmay/alpaca_eval',
        'default_split': 'eval',
        'type': 'alpaca_eval',
    },
    'oasst1': {
        'hf_name': 'OpenAssistant/oasst1',
        'default_split': 'train',
        'type': 'oasst',
    },
    'ultrachat': {
        'hf_name': 'HuggingFaceH4/ultrachat_200k',
        'default_split': 'train_sft',
        'type': 'chat_messages',
    },
    'xsum': {
        'hf_name': 'xsum',
        'default_split': 'train',
        'type': 'summarization',
    },
    'cnn_dailymail': {
        'hf_name': 'cnn_dailymail',
        'hf_kwargs': {'name': '3.0.0'},
        'default_split': 'train',
        'type': 'summarization',
    },
    'ag_news': {
        'hf_name': 'ag_news',
        'default_split': 'train',
        'type': 'classification',
    },
    'banking77': {
        # The original loader is script-based, so use a parquet copy here.
        'hf_name': 'mteb/banking77',
        'default_split': 'train',
        'type': 'classification',
    },
}


def list_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())


def _clip(text: str, max_chars: int = 900) -> str:
    text = ' '.join(text.split())
    return text[:max_chars].strip()


def _join_dialogue(example: Dict, max_turns: int = 8) -> str:
    for key in ('dialog', 'dialogue', 'utterances', 'conversation', 'turns', 'messages'):
        value = example.get(key)
        if isinstance(value, list):
            joined: List[str] = []
            for item in value[:max_turns]:
                if isinstance(item, dict):
                    role = str(item.get('role', item.get('from', 'speaker'))).strip()
                    text = str(item.get('content', item.get('text', item.get('value', '')))).strip()
                    if text:
                        joined.append(f'{role}: {text}')
                else:
                    text = str(item).strip()
                    if text:
                        joined.append(text)
            if joined:
                return ' '.join(joined)
    for key in ('dialog', 'dialogue', 'text', 'prompt', 'content'):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


def _extract_chat_messages(example: Dict, max_turns: int = 8) -> str:
    messages = example.get('messages')
    if isinstance(messages, list):
        parts: List[str] = []
        for item in messages[:max_turns]:
            if not isinstance(item, dict):
                continue
            role = str(item.get('role', 'user')).strip()
            content = str(item.get('content', '')).strip()
            if content:
                parts.append(f'{role}: {content}')
        if parts:
            return '\n'.join(parts)
    prompt = str(example.get('prompt', '')).strip()
    response = str(example.get('response', '')).strip()
    if prompt or response:
        parts = []
        if prompt:
            parts.append(f'user: {prompt}')
        if response:
            parts.append(f'assistant: {response}')
        return '\n'.join(parts)
    return ''


def _row_to_prompt(dataset_type: str, example: Dict) -> str:
    if dataset_type == 'dialogue_summary':
        dialog = _clip(_join_dialogue(example), max_chars=1200)
        if not dialog:
            return ''
        return (
            'System: You are a concise summarization assistant. Produce a faithful 1-2 sentence summary.\n'
            'Task: Summarize the following multi-turn dialogue.\n'
            'Dialogue:\n'
            f'{dialog}\n'
            'Summary:'
        )

    if dataset_type == 'instruction':
        instruction = _clip(str(example.get('instruction', '')).strip(), 300)
        context = _clip(str(example.get('context', '')).strip(), 900)
        if not instruction and not context:
            return ''
        parts: List[str] = [
            'System: You are a careful instruction-following assistant.',
            'Task: Follow the instruction and write a concise helpful answer.',
            f'Instruction: {instruction or "N/A"}',
        ]
        if context:
            parts.append(f'Context: {context}')
        parts.append('Assistant response:')
        return '\n'.join(parts)

    if dataset_type == 'alpaca_eval':
        instruction = _clip(str(example.get('instruction', '')).strip(), 400)
        if not instruction:
            return ''
        parts = [
            'System: You are a strong general assistant.',
            f'Instruction: {instruction}',
            'Assistant response:',
        ]
        return '\n'.join(parts)

    if dataset_type == 'oasst':
        text = _clip(str(example.get('text', '')).strip(), 900)
        role = str(example.get('role', 'user')).strip() or 'user'
        lang = str(example.get('lang', '')).strip()
        if not text:
            return ''
        parts: List[str] = [
            'System: You are a helpful and safe assistant.',
            'Task: Continue the assistant conversation helpfully and safely.',
            f'Conversation role: {role}',
        ]
        if lang:
            parts.append(f'Language: {lang}')
        parts.extend([f'Message: {text}', 'Assistant reply:'])
        return '\n'.join(parts)

    if dataset_type == 'chat_messages':
        convo = _clip(_extract_chat_messages(example), 1400)
        if not convo:
            return ''
        return (
            'System: You are a helpful chat assistant. Continue the conversation naturally and safely.\n'
            'Conversation:\n'
            f'{convo}\n'
            'Assistant:'
        )

    if dataset_type == 'summarization':
        document = _clip(str(example.get('document', example.get('article', ''))).strip(), 1600)
        if not document:
            return ''
        parts: List[str] = [
            'System: You are a news summarization assistant.',
            'Task: Produce a one-sentence factual summary.',
            f'Document: {document}',
            'Predicted summary:',
        ]
        return '\n'.join(parts)

    if dataset_type == 'classification':
        text = _clip(str(example.get('text', example.get('utterance', example.get('input', ''))).strip()), 1000)
        if not text:
            return ''
        parts: List[str] = [
            'System: You are a lightweight classifier.',
            'Task: Read the item and identify its most likely category.',
            f'Item: {text}',
            'Predicted category:',
        ]
        return '\n'.join(parts)

    for key in ('prompt', 'text', 'document', 'content'):
        value = str(example.get(key, '')).strip()
        if value:
            return value
    return ''


def load_public_text_rows(dataset_name: str, split: str, limit: int, seed: int = 42) -> List[Dict]:
    if dataset_name not in DATASET_REGISTRY:
        valid = ', '.join(list_datasets())
        raise ValueError(f'Unknown dataset_name: {dataset_name}. Valid options: {valid}')

    cfg = DATASET_REGISTRY[dataset_name]
    hf_name = cfg['hf_name']
    dataset_type = cfg['type']
    actual_split = split or cfg['default_split']

    load_kwargs = dict(cfg.get('hf_kwargs', {}))
    from datasets import load_dataset

    ds = load_dataset(hf_name, split=actual_split, **load_kwargs)
    if hasattr(ds, 'shuffle'):
        ds = ds.shuffle(seed=seed)

    rows: List[Dict] = []
    for ex in ds:
        prompt = _row_to_prompt(dataset_type, ex)
        if not prompt:
            continue
        rows.append({
            'prompt': prompt,
            'source_dataset': dataset_name,
            'hf_name': hf_name,
            'dataset_type': dataset_type,
        })
        if len(rows) >= limit:
            break

    if not rows:
        sample_keys = list(ds[0].keys()) if len(ds) > 0 else []
        raise RuntimeError(
            f'No usable text rows found for dataset {dataset_name} ({hf_name}) split={actual_split}. '
            f'Sample keys: {sample_keys}'
        )

    return rows
