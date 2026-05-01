from __future__ import annotations

import math
import time
from dataclasses import dataclass
import argparse
import json
from typing import Dict, Iterable, List, Tuple

try:
    from .config_loader import CONFIG
except ImportError:
    class _FallbackConfig:
        def get(self, dotted_path: str, default=None):
            return default

    CONFIG = _FallbackConfig()


@dataclass
class SemanticMatch:
    prefix_tokens: Tuple[int, ...]
    similarity: float
    prefix_len: int
    observations: int
    last_seen: float


class TokenSketcher:
    """Tiny dependency-free semantic-ish sketch over token streams.

    This is deliberately lightweight. It is not a sentence transformer; it is a
    serving-side signal that captures lexical overlap, repeated instruction
    markers, and n-gram shape well enough for policy decisions without adding a
    model dependency to the hot path.
    """

    def __init__(self, dims: int = 128) -> None:
        self.dims = max(int(dims), 16)

    def sketch(self, tokens: Tuple[int, ...]) -> Tuple[float, ...]:
        vec = [0.0] * self.dims
        if not tokens:
            return tuple(vec)
        # unigram + bigram feature hashing. Signs reduce collision bias.
        for i, tok in enumerate(tokens):
            h = (int(tok) * 2654435761) & 0xFFFFFFFF
            idx = h % self.dims
            sign = 1.0 if ((h >> 31) & 1) == 0 else -1.0
            vec[idx] += sign
            if i + 1 < len(tokens):
                bg = ((int(tok) * 1315423911) ^ (int(tokens[i + 1]) * 97531)) & 0xFFFFFFFF
                bidx = bg % self.dims
                bsign = 0.5 if ((bg >> 30) & 1) == 0 else -0.5
                vec[bidx] += bsign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return tuple(x / norm for x in vec)

    @staticmethod
    def cosine(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))


def token_entropy(tokens: Tuple[int, ...]) -> float:
    if not tokens:
        return 0.0
    counts: Dict[int, int] = {}
    for t in tokens:
        counts[int(t)] = counts.get(int(t), 0) + 1
    n = len(tokens)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def longest_common_prefix_len(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


class SemanticKVIndex:
    """In-memory semantic index for observed/cacheable prefixes.

    The index supports two uses:
    1. policy signals: similarity to known request families;
    2. safe exact reuse fallback: closest semantic matches are still checked for
       an exact prefix before real KV reuse is admitted.
    """

    def __init__(self, dims: int = 128, max_entries: int = 512) -> None:
        self.sketcher = TokenSketcher(dims=dims)
        self.max_entries = max_entries
        self._rows: Dict[Tuple[int, ...], dict] = {}

    def add(self, prefix_tokens: Tuple[int, ...], semantic_key: str | None = None) -> None:
        if not prefix_tokens:
            return
        now = time.time()
        row = self._rows.get(prefix_tokens)
        if row is None:
            if len(self._rows) >= self.max_entries:
                oldest = min(self._rows, key=lambda k: self._rows[k]['last_seen'])
                del self._rows[oldest]
            row = {
                'vector': self.sketcher.sketch(prefix_tokens),
                'observations': 0,
                'last_seen': now,
                'semantic_key': semantic_key,
            }
            self._rows[prefix_tokens] = row
        if semantic_key and not row.get('semantic_key'):
            row['semantic_key'] = semantic_key
        row['observations'] = int(row.get('observations', 0)) + 1
        row['last_seen'] = now

    def query(self, tokens: Tuple[int, ...], k: int = 4, min_similarity: float = 0.35, semantic_key: str | None = None) -> List[SemanticMatch]:
        if not tokens or not self._rows:
            return []
        q = self.sketcher.sketch(tokens)
        matches: List[SemanticMatch] = []
        for prefix, row in self._rows.items():
            sim = self.sketcher.cosine(q, row['vector'])
            if semantic_key and row.get('semantic_key') == semantic_key:
                # A paraphrase workload carries an explicit equivalence key for
                # scaffolds that express the same serving task with different
                # surface forms. The boost exposes semantic opportunity without
                # requiring an encoder dependency in the hot path.
                sim = max(sim, float(CONFIG.get('semantic.index.equivalence_key_boost', 0.92)))
            if sim < min_similarity:
                continue
            matches.append(
                SemanticMatch(
                    prefix_tokens=prefix,
                    similarity=sim,
                    prefix_len=len(prefix),
                    observations=int(row.get('observations', 0)),
                    last_seen=float(row.get('last_seen', 0.0)),
                )
            )
        matches.sort(key=lambda m: (m.similarity, m.observations, m.prefix_len), reverse=True)
        return matches[:k]

    def diagnostics(self) -> Dict:
        family_counts: Dict[str, int] = {}
        total_prefix_tokens = 0
        total_observations = 0
        for prefix, row in self._rows.items():
            family = str(row.get('semantic_key') or '<none>')
            family_counts[family] = family_counts.get(family, 0) + 1
            total_prefix_tokens += len(prefix)
            total_observations += int(row.get('observations', 0))
        vector_bytes = len(self._rows) * self.sketcher.dims * 8
        token_bytes = total_prefix_tokens * 8
        return {
            'entries': len(self._rows),
            'max_entries': int(self.max_entries),
            'dims': int(self.sketcher.dims),
            'families': len(family_counts),
            'family_counts': family_counts,
            'total_prefix_tokens': total_prefix_tokens,
            'total_observations': total_observations,
            'approx_memory_bytes': int(vector_bytes + token_bytes),
            'approx_vector_bytes': int(vector_bytes),
            'approx_token_bytes': int(token_bytes),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description='Print SemanticKVIndex diagnostic schema.')
    parser.add_argument('--empty', action='store_true', help='Emit diagnostics for an empty index.')
    args = parser.parse_args()
    index = SemanticKVIndex()
    print(json.dumps(index.diagnostics(), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
