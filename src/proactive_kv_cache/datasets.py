from __future__ import annotations

from typing import Dict, List, Tuple



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

PROMPT_MODES = ('raw', 'templated', 'rag')

SHARED_TEMPLATES = {
    'dialogue_summary': (
        'System: You are serving repeated dialogue summarization traffic in a shared batching environment.\n'
        'Shared editorial policy: preserve speaker intent, compress repetitive turns, keep the summary factual, '
        'and prefer a compact 1-2 sentence answer.\n'
        'Shared execution checklist:\n'
        '1. Read the request payload.\n'
        '2. Identify the dialogue goal and outcome.\n'
        '3. Produce a concise neutral summary.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'instruction': (
        'System: You are serving repeated instruction-following traffic in a shared batching environment.\n'
        'Shared response policy: be helpful, concise, faithful to the provided context, and avoid inventing facts.\n'
        'Shared execution checklist:\n'
        '1. Identify the instruction.\n'
        '2. Use only relevant context.\n'
        '3. Return a short, direct answer.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'alpaca_eval': (
        'System: You are serving evaluation prompts under a shared general-assistant scaffold.\n'
        'Shared response contract: answer directly, keep the tone neutral, and avoid unnecessary verbosity.\n'
        'Shared execution checklist:\n'
        '1. Read the instruction carefully.\n'
        '2. Produce a strong, concise answer.\n'
        '3. Avoid meta commentary.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'oasst': (
        'System: You are serving assistant-turn continuation prompts under a shared safety scaffold.\n'
        'Shared response policy: continue helpfully, respect the stated language, and avoid unsafe instructions.\n'
        'Shared execution checklist:\n'
        '1. Identify the conversation role.\n'
        '2. Continue naturally.\n'
        '3. Keep the answer concise and safe.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'chat_messages': (
        'System: You are serving repeated chat-style prompts in a shared assistant scaffold.\n'
        'Shared response policy: continue the dialogue naturally, preserve user intent, and answer with a helpful next turn.\n'
        'Shared execution checklist:\n'
        '1. Read the dialogue transcript.\n'
        '2. Infer the next helpful assistant step.\n'
        '3. Keep the answer grounded in the conversation.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'summarization': (
        'System: You are serving repeated long-form summarization prompts in a shared editorial scaffold.\n'
        'Shared editorial policy: preserve factual claims, foreground the main event, avoid speculation, and keep the summary compact.\n'
        'Shared execution checklist:\n'
        '1. Read the document payload.\n'
        '2. Extract the central event and supporting context.\n'
        '3. Produce a short factual summary.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'classification': (
        'System: You are serving repeated lightweight classification prompts in a shared decision scaffold.\n'
        'Shared classification policy: focus on the main intent, avoid overthinking, and emit the single most likely label.\n'
        'Shared execution checklist:\n'
        '1. Read the item.\n'
        '2. Identify the dominant intent or topic.\n'
        '3. Produce one category.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'generic': (
        'System: You are serving repeated prompts in a shared assistant scaffold.\n'
        'Shared response policy: stay concise, grounded, and directly useful.\n'
        'Request payload begins below.\n'
    ),
}

RAG_SHARED_TEMPLATES = {
    'dialogue_summary': (
        'System: You are a retrieval-grounded dialogue summarization assistant operating in a batched serving pipeline.\n'
        'Shared evidence protocol:\n'
        '- Use only the provided request payload and the shared rubric below.\n'
        '- Preserve intent, outcome, and speaker actions.\n'
        '- Keep the answer compact and factual.\n'
        'Shared rubric chunk A: summarize the goal.\n'
        'Shared rubric chunk B: summarize the resolution.\n'
        'Shared rubric chunk C: omit filler and greetings.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'instruction': (
        'System: You are a retrieval-grounded instruction-following assistant in a shared serving scaffold.\n'
        'Shared evidence protocol:\n'
        '- Treat the request payload as the source of truth.\n'
        '- Use context only when present.\n'
        '- Keep the answer direct and operationally useful.\n'
        'Shared rubric chunk A: identify the task.\n'
        'Shared rubric chunk B: extract supporting context.\n'
        'Shared rubric chunk C: respond concisely.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'alpaca_eval': (
        'System: You are a retrieval-grounded general assistant in a shared evaluation scaffold.\n'
        'Shared evidence protocol:\n'
        '- Answer the instruction directly.\n'
        '- Prefer concise reasoning over long exposition.\n'
        '- Avoid hedging unless uncertainty is explicit.\n'
        'Shared rubric chunk A: interpret the task.\n'
        'Shared rubric chunk B: answer clearly.\n'
        'Shared rubric chunk C: avoid filler.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'oasst': (
        'System: You are a retrieval-grounded assistant-turn generator under a shared safety scaffold.\n'
        'Shared evidence protocol:\n'
        '- Respect the conversation role and language.\n'
        '- Continue helpfully without contradicting prior context.\n'
        '- Keep the answer safe and compact.\n'
        'Shared rubric chunk A: identify the user need.\n'
        'Shared rubric chunk B: respond safely.\n'
        'Shared rubric chunk C: continue naturally.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'chat_messages': (
        'System: You are a retrieval-grounded chat assistant operating with a shared conversation scaffold.\n'
        'Shared evidence protocol:\n'
        '- Use the dialogue transcript as the primary context.\n'
        '- Preserve topic continuity.\n'
        '- Produce the next helpful assistant turn only.\n'
        'Shared rubric chunk A: read the last user need.\n'
        'Shared rubric chunk B: continue naturally.\n'
        'Shared rubric chunk C: stay concise.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'summarization': (
        'System: You are a retrieval-grounded news summarization assistant in a shared serving scaffold.\n'
        'Shared evidence protocol:\n'
        '- Use only the provided document payload.\n'
        '- Preserve factual claims and the central event.\n'
        '- Keep the answer to one compact summary sentence.\n'
        'Shared rubric chunk A: identify the main event.\n'
        'Shared rubric chunk B: capture key supporting context.\n'
        'Shared rubric chunk C: avoid speculation and adjectives.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'classification': (
        'System: You are a retrieval-grounded classifier operating inside a shared decision scaffold.\n'
        'Shared evidence protocol:\n'
        '- Focus on the main intent signal.\n'
        '- Ignore incidental wording.\n'
        '- Emit the single strongest category.\n'
        'Shared rubric chunk A: identify the dominant topic.\n'
        'Shared rubric chunk B: resolve ambiguity conservatively.\n'
        'Shared rubric chunk C: answer with one label.\n'
        'Dataset family: {dataset_label}.\n'
        'Request payload begins below.\n'
    ),
    'generic': (
        'System: You are a retrieval-grounded assistant operating inside a shared serving scaffold.\n'
        'Shared evidence protocol:\n'
        '- Use the request payload as the source of truth.\n'
        '- Answer directly and concisely.\n'
        'Request payload begins below.\n'
    ),
}


def list_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())


def list_prompt_modes() -> List[str]:
    return list(PROMPT_MODES)


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


def _resolve_prompt_mode(prompt_mode: str) -> str:
    if prompt_mode not in PROMPT_MODES:
        valid = ', '.join(PROMPT_MODES)
        raise ValueError(f'Unknown prompt_mode: {prompt_mode}. Valid options: {valid}')
    return prompt_mode


def _apply_prompt_mode(dataset_name: str, dataset_type: str, base_prompt: str, prompt_mode: str) -> Tuple[str, str]:
    prompt_mode = _resolve_prompt_mode(prompt_mode)
    if prompt_mode == 'raw':
        return base_prompt, ''
    template_bank = RAG_SHARED_TEMPLATES if prompt_mode == 'rag' else SHARED_TEMPLATES
    dataset_label = dataset_name.replace('_', ' ')
    shared_prefix = template_bank.get(dataset_type, template_bank['generic']).format(dataset_label=dataset_label)
    return f'{shared_prefix}{base_prompt}', shared_prefix


def load_public_text_rows(dataset_name: str, split: str, limit: int, seed: int = 42, prompt_mode: str = 'raw') -> List[Dict]:
    if dataset_name not in DATASET_REGISTRY:
        valid = ', '.join(list_datasets())
        raise ValueError(f'Unknown dataset_name: {dataset_name}. Valid options: {valid}')

    cfg = DATASET_REGISTRY[dataset_name]
    hf_name = cfg['hf_name']
    dataset_type = cfg['type']
    actual_split = split or cfg['default_split']
    resolved_prompt_mode = _resolve_prompt_mode(prompt_mode)

    load_kwargs = dict(cfg.get('hf_kwargs', {}))
    from datasets import load_dataset

    ds = load_dataset(hf_name, split=actual_split, **load_kwargs)
    if hasattr(ds, 'shuffle'):
        ds = ds.shuffle(seed=seed)

    rows: List[Dict] = []
    for ex in ds:
        base_prompt = _row_to_prompt(dataset_type, ex)
        if not base_prompt:
            continue
        prompt, shared_prefix_text = _apply_prompt_mode(dataset_name, dataset_type, base_prompt, resolved_prompt_mode)
        rows.append({
            'prompt': prompt,
            'source_dataset': dataset_name,
            'hf_name': hf_name,
            'dataset_type': dataset_type,
            'prompt_mode': resolved_prompt_mode,
            'shared_prefix_text': shared_prefix_text,
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
