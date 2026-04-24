from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .utils import common_prefix_length


@dataclass
class CacheEntry:
    prefix_tokens: Tuple[int, ...]
    kv_cache: Any
    frequency: float
    last_access: float
    generation_cost_ms: float
    memory_bytes: int
    tier: str = 'cpu'
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    was_speculative: bool = False
    speculative_reused: bool = False
    first_reuse_at: float | None = None

    def utility_score(self, now: float, recency_weight: float = 0.25) -> float:
        age = max(now - self.last_access, 0.0)
        recency = 1.0 / (1.0 + age / 30.0)
        effective_len = min(len(self.prefix_tokens), 32)
        savings_density = (self.frequency * self.generation_cost_ms * max(effective_len, 1)) / max(self.memory_bytes, 1)
        tier_bonus = 0.05 if self.tier == 'gpu' else 0.0
        speculative_penalty = 0.05 if self.was_speculative and not self.speculative_reused else 0.0
        short_prefix_bonus = 0.03 if len(self.prefix_tokens) <= 16 else 0.0
        return (1.0 - recency_weight) * savings_density + recency_weight * recency + tier_bonus + short_prefix_bonus - speculative_penalty


class TieredStateBank:
    def __init__(self, max_memory_bytes: int, ema_alpha: float = 0.2, min_match_length: int = 3):
        self.max_memory_bytes = max_memory_bytes
        self.ema_alpha = ema_alpha
        self.min_match_length = min_match_length
        self.entries: Dict[Tuple[int, ...], CacheEntry] = {}
        self.frequency_counter: Dict[Tuple[int, ...], float] = defaultdict(float)
        self.observation_count: Dict[Tuple[int, ...], int] = defaultdict(int)
        self.current_memory_bytes = 0
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'speculative_hits': 0,
            'evictions': 0,
            'wasted_precomputes': 0,
            'wasted_compute_ms': 0.0,
            'useful_speculative_savings_ms': 0.0,
            'promotions': 0,
            'demotions': 0,
            'bypassed_matches': 0,
            'reuse_failures': 0,
        }
        self._lock = threading.RLock()

    def add_metric(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.metrics[name] = self.metrics.get(name, 0.0) + value

    def default_prefix_lengths(self, tokens: Tuple[int, ...]) -> List[int]:
        return self._default_prefix_lengths(tokens)

    def contains(self, prefix_tokens: Tuple[int, ...]) -> bool:
        with self._lock:
            return prefix_tokens in self.entries

    def get_memory_bytes(self) -> int:
        with self._lock:
            return self.current_memory_bytes

    def get_frequency(self, prefix_tokens: Tuple[int, ...]) -> float:
        with self._lock:
            return float(self.frequency_counter.get(prefix_tokens, 0.0))

    def get_observation_count(self, prefix_tokens: Tuple[int, ...]) -> int:
        with self._lock:
            return int(self.observation_count.get(prefix_tokens, 0))

    def get_entry(self, prefix_tokens: Tuple[int, ...]) -> CacheEntry | None:
        with self._lock:
            return self.entries.get(prefix_tokens)

    def get_candidate_frequencies(self, max_prefix_len: int | None = None, exclude_stored: bool = True) -> List[Tuple[Tuple[int, ...], float]]:
        with self._lock:
            items: List[Tuple[Tuple[int, ...], float]] = []
            for prefix, freq in self.frequency_counter.items():
                if exclude_stored and prefix in self.entries:
                    continue
                if max_prefix_len is not None and len(prefix) > max_prefix_len:
                    continue
                items.append((prefix, float(freq)))
            return items

    def get_candidate_stats(self, max_prefix_len: int | None = None, exclude_stored: bool = True) -> List[Tuple[Tuple[int, ...], float, int]]:
        with self._lock:
            items: List[Tuple[Tuple[int, ...], float, int]] = []
            for prefix, freq in self.frequency_counter.items():
                if exclude_stored and prefix in self.entries:
                    continue
                if max_prefix_len is not None and len(prefix) > max_prefix_len:
                    continue
                items.append((prefix, float(freq), int(self.observation_count.get(prefix, 0))))
            return items

    def observe_query(self, tokens: Tuple[int, ...], tracked_prefix_lengths: Optional[List[int]] = None) -> None:
        tracked_prefix_lengths = tracked_prefix_lengths or self._default_prefix_lengths(tokens)
        with self._lock:
            for length in tracked_prefix_lengths:
                if length < self.min_match_length or length > len(tokens):
                    continue
                prefix = tokens[:length]
                old = self.frequency_counter[prefix]
                self.frequency_counter[prefix] = self.ema_alpha + (1.0 - self.ema_alpha) * old
                self.observation_count[prefix] += 1

    def peek_match(self, tokens: Tuple[int, ...]) -> Optional[Tuple[Tuple[int, ...], CacheEntry, int]]:
        with self._lock:
            return self._find_match_unlocked(tokens)

    def record_miss(self) -> None:
        self.add_metric('misses', 1)

    def admit_match(self, prefix_tokens: Tuple[int, ...]) -> Optional[CacheEntry]:
        with self._lock:
            entry = self.entries.get(prefix_tokens)
            if entry is None:
                return None
            entry.last_access = time.time()
            entry.hit_count += 1
            entry.frequency = min(1.0, (1.0 - self.ema_alpha) * entry.frequency + self.ema_alpha)
            self.metrics['hits'] += 1
            if entry.was_speculative:
                self.metrics['speculative_hits'] += 1
                if not entry.speculative_reused:
                    entry.speculative_reused = True
                    entry.first_reuse_at = time.time()
                self.metrics['useful_speculative_savings_ms'] += float(entry.generation_cost_ms)
            return entry

    def lookup(self, tokens: Tuple[int, ...]) -> Optional[Tuple[Tuple[int, ...], CacheEntry, int]]:
        match = self.peek_match(tokens)
        if match is None:
            self.record_miss()
            return None
        key, _, match_len = match
        entry = self.admit_match(key)
        if entry is None:
            self.record_miss()
            return None
        return key, entry, match_len

    def store(
        self,
        prefix_tokens: Tuple[int, ...],
        kv_cache: Any,
        generation_cost_ms: float,
        memory_bytes: int,
        is_speculative: bool,
        tier: str = 'cpu',
    ) -> bool:
        if len(prefix_tokens) < self.min_match_length:
            return False

        with self._lock:
            if prefix_tokens in self.entries:
                self._remove_unlocked(prefix_tokens)

            while self.current_memory_bytes + memory_bytes > self.max_memory_bytes:
                if not self._evict_one_unlocked():
                    return False

            entry = CacheEntry(
                prefix_tokens=prefix_tokens,
                kv_cache=kv_cache,
                frequency=self.frequency_counter.get(prefix_tokens, 0.02),
                last_access=time.time(),
                generation_cost_ms=generation_cost_ms,
                memory_bytes=memory_bytes,
                tier=tier,
                was_speculative=is_speculative,
                speculative_reused=False,
                first_reuse_at=None,
            )
            self.entries[prefix_tokens] = entry
            self.current_memory_bytes += memory_bytes
            return True

    def promote(self, prefix_tokens: Tuple[int, ...], new_cache: Any, new_memory_bytes: int, new_tier: str = 'gpu') -> bool:
        with self._lock:
            entry = self.entries.get(prefix_tokens)
            if entry is None:
                return False

            old_size = entry.memory_bytes
            while self.current_memory_bytes - old_size + new_memory_bytes > self.max_memory_bytes:
                if not self._evict_one_unlocked(exclude_key=prefix_tokens):
                    return False

            self.current_memory_bytes = self.current_memory_bytes - old_size + new_memory_bytes
            entry.kv_cache = new_cache
            entry.memory_bytes = new_memory_bytes
            if entry.tier != new_tier:
                self.metrics['promotions'] += 1
            entry.tier = new_tier
            entry.last_access = time.time()
            return True

    def demote(self, prefix_tokens: Tuple[int, ...], new_cache: Any, new_memory_bytes: int, new_tier: str = 'cpu') -> bool:
        with self._lock:
            entry = self.entries.get(prefix_tokens)
            if entry is None:
                return False

            self.current_memory_bytes = self.current_memory_bytes - entry.memory_bytes + new_memory_bytes
            entry.kv_cache = new_cache
            entry.memory_bytes = new_memory_bytes
            if entry.tier != new_tier:
                self.metrics['demotions'] += 1
            entry.tier = new_tier
            entry.last_access = time.time()
            return True

    def remove(self, prefix_tokens: Tuple[int, ...]) -> None:
        with self._lock:
            self._remove_unlocked(prefix_tokens)

    def finalize_run(self) -> None:
        with self._lock:
            for entry in self.entries.values():
                if entry.was_speculative and not entry.speculative_reused:
                    self.metrics['wasted_precomputes'] += 1
                    self.metrics['wasted_compute_ms'] += float(entry.generation_cost_ms)
                    entry.speculative_reused = True

    def top_candidates(self, k: int, exclude_stored: bool = True, max_prefix_len: int | None = 24) -> List[Tuple[int, ...]]:
        with self._lock:
            items = []
            for prefix, freq in self.frequency_counter.items():
                if exclude_stored and prefix in self.entries:
                    continue
                if max_prefix_len is not None and len(prefix) > max_prefix_len:
                    continue
                effective_len = min(len(prefix), 16)
                score = float(freq) * (1.0 + 0.08 * effective_len)
                items.append((prefix, score))
            items.sort(key=lambda x: (-x[1], len(x[0])))
            return [prefix for prefix, _ in items[:k]]

    def snapshot_metrics(self) -> Dict[str, float]:
        with self._lock:
            total = self.metrics['hits'] + self.metrics['misses']
            tier_counts = {'gpu_entries': 0, 'cpu_entries': 0}
            for entry in self.entries.values():
                tier_counts[f'{entry.tier}_entries'] = tier_counts.get(f'{entry.tier}_entries', 0) + 1
            return {
                **self.metrics,
                **tier_counts,
                'entries_stored': len(self.entries),
                'memory_used_mb': self.current_memory_bytes / (1024 ** 2),
                'hit_rate': self.metrics['hits'] / max(total, 1),
                'speculative_hit_rate': self.metrics['speculative_hits'] / max(self.metrics['hits'], 1),
            }

    def _find_match_unlocked(self, tokens: Tuple[int, ...]) -> Optional[Tuple[Tuple[int, ...], CacheEntry, int]]:
        best_key = None
        best_len = self.min_match_length - 1
        for key in self.entries:
            overlap = common_prefix_length(key, tokens)
            if overlap == len(key) and overlap > best_len:
                best_key = key
                best_len = overlap
        if best_key is None:
            return None
        return best_key, self.entries[best_key], best_len

    def _remove_unlocked(self, prefix_tokens: Tuple[int, ...]) -> None:
        entry = self.entries.pop(prefix_tokens, None)
        if entry is None:
            return
        self.current_memory_bytes -= entry.memory_bytes
        if entry.was_speculative and not entry.speculative_reused:
            self.metrics['wasted_precomputes'] += 1
            self.metrics['wasted_compute_ms'] += float(entry.generation_cost_ms)

    def _evict_one_unlocked(self, exclude_key: Tuple[int, ...] | None = None) -> bool:
        keys = [k for k in self.entries if k != exclude_key]
        if not keys:
            return False
        now = time.time()
        worst_key = min(keys, key=lambda k: self.entries[k].utility_score(now))
        self._remove_unlocked(worst_key)
        self.metrics['evictions'] += 1
        return True

    def _default_prefix_lengths(self, tokens: Tuple[int, ...]) -> List[int]:
        n = len(tokens)
        if n <= self.min_match_length:
            return [n]
        lengths = {self.min_match_length, min(n, 4), min(n, 6), min(n, 8), min(n, 12), min(n, 16), min(n, 24), min(n, 32), n}
        return sorted(length for length in lengths if self.min_match_length <= length <= n)
