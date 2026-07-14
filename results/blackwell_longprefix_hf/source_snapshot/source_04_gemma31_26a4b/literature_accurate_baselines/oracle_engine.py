from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

from proactive_kv_cache.engines import ReactivePrefixCacheEngine


class OracleFutureReuseEngine(ReactivePrefixCacheEngine):
    """Offline future-aware exact-prefix oracle kept outside the main harness."""

    def __init__(
        self,
        backend,
        request_trace: Sequence[Tuple[Tuple[int, ...], Dict | None]],
        max_memory_mb: int = 256,
    ) -> None:
        super().__init__(backend=backend, max_memory_mb=max_memory_mb)
        self.name = "oracle_future_reuse"
        self.request_trace = [(tuple(tokens), dict(metadata or {})) for tokens, metadata in request_trace]
        self._active_request_id = -1

    def serve_tokens(self, request_id: int, tokens: Tuple[int, ...], metadata: Dict | None = None):
        self._active_request_id = int(request_id)
        return super().serve_tokens(request_id, tokens, metadata=metadata)

    def _future_match_indices(self, prefix_tokens: Tuple[int, ...], from_request_id: int) -> List[int]:
        indices: List[int] = []
        for idx in range(from_request_id + 1, len(self.request_trace)):
            future_tokens, _ = self.request_trace[idx]
            if len(future_tokens) >= len(prefix_tokens) and future_tokens[: len(prefix_tokens)] == prefix_tokens:
                indices.append(idx)
        return indices

    def _future_utility_ms(self, prefix_tokens: Tuple[int, ...], from_request_id: int) -> float:
        utility = 0.0
        for idx in self._future_match_indices(prefix_tokens, from_request_id):
            future_tokens, future_meta = self.request_trace[idx]
            total_tokens = len(future_tokens)
            match_len = len(prefix_tokens)
            suffix_len = max(total_tokens - match_len, 0)
            scaffold_match = self._has_scaffold_hint(future_tokens, future_meta)
            estimate = (
                self._estimate_scaffold_saved_ms(total_tokens, match_len, suffix_len)
                if scaffold_match
                else self._estimate_saved_ms(total_tokens, match_len, suffix_len)
            )
            utility += max(float(estimate), 0.0)
        return utility

    def _next_future_use(self, prefix_tokens: Tuple[int, ...], from_request_id: int) -> int | None:
        matches = self._future_match_indices(prefix_tokens, from_request_id)
        return matches[0] if matches else None

    def _oracle_make_room(self, request_id: int, required_bytes: int) -> bool:
        if required_bytes > self.bank.max_memory_bytes:
            return False
        while self.bank.get_memory_bytes() + required_bytes > self.bank.max_memory_bytes:
            victims = []
            now = time.time()
            for key, entry in list(self.bank.entries.items()):
                next_use = self._next_future_use(key, request_id)
                remaining_utility = self._future_utility_ms(key, request_id)
                utility_density = remaining_utility / max(entry.memory_bytes, 1)
                victims.append((next_use is None, next_use if next_use is not None else float("inf"), utility_density, key, now - entry.last_access))
            if not victims:
                return False
            victims.sort(key=lambda item: (-int(item[0]), item[1], item[2], -item[4]))
            self.bank.remove(victims[0][3])
        return True

    def _should_store_reactive_prefix(self, tokens: Tuple[int, ...], prefix_len: int, metadata: Dict | None = None) -> bool:
        self.engine_metrics["store_attempts"] += 1
        if not getattr(self.backend, "supports_external_kv", True):
            self.engine_metrics["store_skips"] += 1
            return False
        if prefix_len < self.tuning.min_store_prefix_tokens or prefix_len > self._request_cacheable_prefix_limit(tokens, metadata):
            self.engine_metrics["store_skips"] += 1
            return False
        prefix_tokens = tokens[:prefix_len]
        future_matches = self._future_match_indices(prefix_tokens, self._active_request_id)
        if not future_matches:
            self.engine_metrics["store_skips"] += 1
            return False
        estimated_store_cost = max(self._estimate_full_cost_ms(prefix_len), self.backend.estimate_prefill_cost_ms(prefix_len))
        future_utility = self._future_utility_ms(prefix_tokens, self._active_request_id)
        if future_utility <= estimated_store_cost:
            self.engine_metrics["store_skips"] += 1
            return False
        required_bytes = int(self.backend.estimate_kv_cache_bytes(prefix_len))
        if not self._oracle_make_room(self._active_request_id, required_bytes):
            self.engine_metrics["store_skips"] += 1
            return False
        return True

    def _should_attempt_cache_use(self, tokens: Tuple[int, ...], entry, match_len: int, metadata: Dict | None = None) -> bool:
        if not self.cache_enabled:
            self.engine_metrics["bypassed_matches"] += 1
            self.bank.add_metric("bypassed_matches", 1)
            return False
        if not getattr(self.backend, "supports_external_kv", True):
            self.engine_metrics["bypassed_matches"] += 1
            self.bank.add_metric("bypassed_matches", 1)
            return False
        if match_len < self.bank.min_match_length:
            self.engine_metrics["bypassed_matches"] += 1
            self.bank.add_metric("bypassed_matches", 1)
            return False
        self.engine_metrics["reuse_attempts"] += 1
        total_tokens = len(tokens)
        suffix_len = max(total_tokens - match_len, 0)
        scaffold_match = self._has_scaffold_hint(tokens, metadata)
        estimate = (
            self._estimate_scaffold_saved_ms(total_tokens, match_len, suffix_len)
            if scaffold_match
            else self._estimate_saved_ms(total_tokens, match_len, suffix_len)
        )
        self.engine_metrics["estimated_tokens_saved_total"] = int(self.engine_metrics.get("estimated_tokens_saved_total", 0)) + int(match_len)
        self.engine_metrics["saved_latency_estimate_ms"] = float(self.engine_metrics.get("saved_latency_estimate_ms", 0.0)) + float(max(estimate, 0.0))
        return True
