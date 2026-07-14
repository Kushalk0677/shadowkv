from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

SYSTEM_PROMPTS = [
    'You are a concise assistant. Answer: ',
    'You are a code reviewer. Review: ',
    'Summarize the following text: ',
    'Translate to French: ',
    'Explain simply: ',
    'You are a financial analyst. Interpret: ',
    'Given the context, answer the question: ',
    'Continue this story: ',
]

LONG_SHARED_TEMPLATES = [
    'System: You are an enterprise banking assistant operating in a regulated environment. Always cite the provided policy snippet, summarize customer intent, identify operational risk, and give a short recommended next action.\nPolicy excerpt: KYC refresh is mandatory after trigger events, suspicious patterns require escalation, and customer communications must avoid promising account actions before identity verification.\nUser case:\n',
    'System: You are an insurance claims analyst. Extract the core facts, classify the claim, identify missing evidence, and produce a concise triage note.\nTemplate checklist: incident date, location, claimant statement, policy coverage, exclusions, fraud signals, and next best action.\nClaim narrative:\n',
    'System: You are a retrieval-augmented assistant. Use the provided knowledge chunks to answer faithfully. First restate the question, then synthesize evidence, then answer.\nKnowledge chunks:\nChunk A: The deployment requires a canary, health checks, rollback guardrails, and post-deploy metrics review.\nChunk B: Sensitive data must be minimized, redacted, and audited before persistence.\nChunk C: Incident timelines should capture trigger, impact, containment, remediation, and prevention actions.\nQuestion:\n',
]

QUERY_SUFFIXES = [
    'What is the capital of Australia?',
    'How does photosynthesis work?',
    'The quarterly revenue rose 12 percent while retention fell.',
    'Write a Python function to reverse a linked list.',
    'Once upon a time there was a small robot in a crowded city.',
    'Bonjour, comment allez-vous aujourd hui?',
    'Why is the sky blue?',
    'Explain quantum entanglement in simple terms.',
    'What are the major themes of Hamlet?',
    'The model improved on the validation set but not the test set.',
]

RAG_QUESTIONS = [
    'How should the incident timeline be structured for audit review?',
    'What deployment controls must be present before production rollout?',
    'How should sensitive data be handled before persistence?',
    'What are the mandatory controls for a regulated customer flow?',
]

SEMANTIC_PARAPHRASE_PREFIXES = [
    'System: Classify the following item by dominant topic. Return one short label.\nItem:\n',
    'Classification task: identify the best category for this text. Reply with the label only.\nText:\n',
    'Decision brief: read the item and choose its strongest topic or intent category.\nInput:\n',
    'Task: assign the most likely category to the content below. Keep the answer compact.\nRequest body:\n',
]

SEMANTIC_PARAPHRASE_ITEMS = [
    'The central bank kept rates unchanged after inflation cooled for the third month.',
    'A new battery design helped the electric vehicle travel farther on a single charge.',
    'The football club signed a young striker before the transfer window closed.',
    'Researchers released a compact language model for document summarization.',
    'Shares rose after the company reported stronger quarterly earnings.',
    'A hospital trial showed improved recovery times for the new treatment pathway.',
]

CUSTOMER_CASES = [
    'Customer moved to a new address and wants to raise the transfer limit immediately.',
    'Customer reported a suspicious login and wants the account unlocked today.',
    'Claimant says the accident happened in heavy rain and no third-party witness is available.',
    'Operations team wants to skip the approval gate because the release is urgent.',
]


@dataclass
class Request:
    request_id: int
    prompt: str
    arrival_time: float
    metadata: Optional[dict] = None


class SyntheticWorkloadGenerator:
    def __init__(
        self,
        alpha: float,
        mean_inter_arrival_ms: float,
        burst_probability: float = 0.0,
        burst_size: int = 6,
        seed: int = 42,
        hot_suffix_bias: float = 0.0,
        long_prefix_bias: float = 0.0,
        rag_mode: bool = False,
        semantic_mode: bool = False,
    ):
        self.alpha = alpha
        self.mean_inter_arrival_ms = mean_inter_arrival_ms
        self.burst_probability = burst_probability
        self.burst_size = burst_size
        self.hot_suffix_bias = hot_suffix_bias
        self.long_prefix_bias = long_prefix_bias
        self.rag_mode = rag_mode
        self.semantic_mode = semantic_mode
        self.rng = np.random.default_rng(seed)

        ranks = np.arange(1, len(SYSTEM_PROMPTS) + 1)
        probs = 1.0 / (ranks ** alpha)
        self.template_probs = probs / probs.sum()

        suffix_ranks = np.arange(1, len(QUERY_SUFFIXES) + 1)
        suffix_probs = 1.0 / (suffix_ranks ** max(alpha, 0.05))
        if hot_suffix_bias > 0:
            suffix_probs = suffix_probs ** (1.0 + hot_suffix_bias)
        self.suffix_probs = suffix_probs / suffix_probs.sum()

        long_ranks = np.arange(1, len(LONG_SHARED_TEMPLATES) + 1)
        long_probs = 1.0 / (long_ranks ** max(alpha + long_prefix_bias, 0.2))
        self.long_template_probs = long_probs / long_probs.sum()

    def generate(self, n_requests: int) -> List[Request]:
        if n_requests <= 0:
            raise ValueError('n_requests must be positive')
        current_time = 0.0
        rows: List[Request] = []
        i = 0

        while i < n_requests:
            if self.burst_probability > 0 and self.rng.random() < self.burst_probability:
                burst_len = min(self.burst_size, n_requests - i)
                for _ in range(burst_len):
                    prompt, metadata = self._sample_prompt(i)
                    rows.append(Request(request_id=i, prompt=prompt, arrival_time=current_time, metadata=metadata))
                    current_time += 0.008
                    i += 1
                current_time += float(self.rng.exponential(self.mean_inter_arrival_ms * 4) / 1000.0)
                continue

            prompt, metadata = self._sample_prompt(i)
            rows.append(Request(request_id=i, prompt=prompt, arrival_time=current_time, metadata=metadata))
            current_time += float(self.rng.exponential(self.mean_inter_arrival_ms) / 1000.0)
            i += 1

        return rows

    def _sample_prompt(self, request_id: int) -> Tuple[str, Dict[str, object]]:
        if self.semantic_mode:
            variant = request_id % len(SEMANTIC_PARAPHRASE_PREFIXES)
            item = str(self.rng.choice(SEMANTIC_PARAPHRASE_ITEMS))
            prompt = f"{SEMANTIC_PARAPHRASE_PREFIXES[variant]}{item}\nCategory:"
            return prompt, {
                'source_workload': 'synthetic',
                'variant': 'semantic_paraphrase',
                'prompt_mode': 'semantic',
                'shared_prefix_text': SEMANTIC_PARAPHRASE_PREFIXES[variant],
                'semantic_equivalence_key': 'synthetic_semantic_classification',
                'semantic_family': 'classification',
                'paraphrase_variant': variant,
                'semantic_variant_count': len(SEMANTIC_PARAPHRASE_PREFIXES),
            }
        if self.long_prefix_bias > 0 or self.rag_mode:
            t = int(self.rng.choice(len(LONG_SHARED_TEMPLATES), p=self.long_template_probs))
            case = str(self.rng.choice(CUSTOMER_CASES))
            shared_prefix = LONG_SHARED_TEMPLATES[t]
            if self.rag_mode:
                question = str(self.rng.choice(RAG_QUESTIONS))
                variant_bucket = request_id % 5
                prompt = f"{shared_prefix}{question}\nContext case: {case}\nVariant {variant_bucket}."
                return prompt, {
                    'source_workload': 'synthetic',
                    'variant': 'rag_long_context',
                    'prompt_mode': 'rag',
                    'shared_prefix_text': shared_prefix,
                }
            variant_bucket = request_id % 7
            suffix = str(self.rng.choice(QUERY_SUFFIXES, p=self.suffix_probs))
            prompt = f"{shared_prefix}{case} {suffix} Variant {variant_bucket}."
            return prompt, {
                'source_workload': 'synthetic',
                'variant': 'long_shared_prefix',
                'prompt_mode': 'templated',
                'shared_prefix_text': shared_prefix,
            }

        t = int(self.rng.choice(len(SYSTEM_PROMPTS), p=self.template_probs))
        s = int(self.rng.choice(len(QUERY_SUFFIXES), p=self.suffix_probs))
        variant_bucket = request_id % 7
        shared_prefix = SYSTEM_PROMPTS[t]
        prompt = f"{shared_prefix}{QUERY_SUFFIXES[s]} Variant {variant_bucket}."
        return prompt, {
            'source_workload': 'synthetic',
            'variant': 'standard',
            'prompt_mode': 'templated',
            'shared_prefix_text': shared_prefix,
        }


SYNTHETIC_VARIANTS = {
    'uniform': dict(alpha=0.2, mean_inter_arrival_ms=80.0, burst_probability=0.0, hot_suffix_bias=0.0),
    'mild_skew': dict(alpha=0.8, mean_inter_arrival_ms=120.0, burst_probability=0.0, hot_suffix_bias=0.1),
    'moderate_skew': dict(alpha=1.2, mean_inter_arrival_ms=150.0, burst_probability=0.0, hot_suffix_bias=0.2),
    'high_skew': dict(alpha=2.0, mean_inter_arrival_ms=200.0, burst_probability=0.0, hot_suffix_bias=0.4),
    'bursty_mild': dict(alpha=1.0, mean_inter_arrival_ms=180.0, burst_probability=0.25, hot_suffix_bias=0.15),
    'bursty_high': dict(alpha=1.5, mean_inter_arrival_ms=250.0, burst_probability=0.4, hot_suffix_bias=0.3),
    'low_skew': dict(alpha=0.35, mean_inter_arrival_ms=90.0, burst_probability=0.0, hot_suffix_bias=0.0),
    'mixed': dict(alpha=1.0, mean_inter_arrival_ms=130.0, burst_probability=0.15, hot_suffix_bias=0.15),
    'long_shared_prefix': dict(alpha=1.6, mean_inter_arrival_ms=180.0, burst_probability=0.15, hot_suffix_bias=0.2, long_prefix_bias=1.0),
    'rag_long_context': dict(alpha=1.7, mean_inter_arrival_ms=200.0, burst_probability=0.10, hot_suffix_bias=0.1, long_prefix_bias=1.2, rag_mode=True),
    'semantic_paraphrase': dict(alpha=1.0, mean_inter_arrival_ms=120.0, burst_probability=0.0, hot_suffix_bias=0.0, semantic_mode=True),
}


def make_synthetic_workload(variant: str, n_requests: int, seed: int = 42, mean_inter_arrival_ms: float | None = None) -> List[Request]:
    if n_requests <= 0:
        raise ValueError('n_requests must be positive')
    if variant not in SYNTHETIC_VARIANTS:
        valid = ', '.join(sorted(SYNTHETIC_VARIANTS.keys()))
        raise ValueError(f'Unknown variant: {variant}. Valid variants: {valid}')
    settings = dict(SYNTHETIC_VARIANTS[variant])
    if mean_inter_arrival_ms is not None:
        settings['mean_inter_arrival_ms'] = mean_inter_arrival_ms
    generator = SyntheticWorkloadGenerator(seed=seed, **settings)
    return generator.generate(n_requests)


def make_public_dataset_workload(
    dataset_name: str,
    split: str,
    n_requests: int,
    seed: int = 42,
    mean_inter_arrival_ms: float = 150.0,
    prompt_mode: str = 'raw',
) -> List[Request]:
    if n_requests <= 0:
        raise ValueError('n_requests must be positive')
    from .datasets import load_public_text_rows

    rows = load_public_text_rows(dataset_name=dataset_name, split=split, limit=n_requests, seed=seed, prompt_mode=prompt_mode)
    rng = np.random.default_rng(seed)
    current = 0.0
    requests: List[Request] = []

    for i, row in enumerate(rows):
        requests.append(Request(request_id=i, prompt=row['prompt'], arrival_time=current, metadata=row))
        current += float(rng.exponential(mean_inter_arrival_ms) / 1000.0)

    return requests
