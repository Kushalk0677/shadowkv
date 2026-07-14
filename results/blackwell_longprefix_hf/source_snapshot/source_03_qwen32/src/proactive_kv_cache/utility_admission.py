from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


def prefix_length_bucket(prefix_tokens: int) -> str:
    n = max(int(prefix_tokens), 0)
    for upper in (64, 128, 256, 512, 1024, 2048, 4096):
        if n <= upper:
            lower = 0 if upper == 64 else ({128: 65, 256: 129, 512: 257, 1024: 513, 2048: 1025, 4096: 2049}[upper])
            return f"{lower}-{upper}"
    return "4097+"


@dataclass
class BucketStats:
    observations: int = 0
    ewma_full_ms_per_token: float = 0.0
    ewma_reuse_overhead_ms: float = 0.0
    ewma_reuse_ms_per_suffix_token: float = 0.0
    admitted: int = 0
    bypassed_negative: int = 0
    alpha: float = 0.20

    def _update_ewma(self, current: float, observed: float) -> float:
        observed = float(max(observed, 0.0))
        if current <= 0.0:
            return observed
        return (1.0 - self.alpha) * current + self.alpha * observed

    def update_full(self, token_count: int, latency_ms: float) -> None:
        if token_count <= 0:
            return
        self.observations += 1
        self.ewma_full_ms_per_token = self._update_ewma(self.ewma_full_ms_per_token, latency_ms / max(token_count, 1))

    def update_reuse(self, suffix_tokens: int, latency_ms: float, matched_prefix_tokens: int, fallback_full_ms_per_token: float) -> None:
        self.observations += 1
        if suffix_tokens > 0:
            self.ewma_reuse_ms_per_suffix_token = self._update_ewma(
                self.ewma_reuse_ms_per_suffix_token,
                latency_ms / max(suffix_tokens, 1),
            )
        estimated_suffix_cost = max(float(fallback_full_ms_per_token), 1e-9) * max(int(suffix_tokens), 0)
        inferred_overhead = max(float(latency_ms) - estimated_suffix_cost, 0.0)
        self.ewma_reuse_overhead_ms = self._update_ewma(self.ewma_reuse_overhead_ms, inferred_overhead)


@dataclass
class UtilityDecision:
    admit: bool
    bucket: str
    expected_saved_ms: float
    expected_cost_ms: float
    net_utility_ms: float
    reason: str


@dataclass
class OnlineUtilityEstimator:
    """Low-overhead bucketed estimator for exact-prefix admission.

    It learns two quantities online: approximate full-prefill cost per token and
    reuse overhead. Admission is based on expected net utility, not hit rate.
    """

    default_full_ms_per_token: float = 0.20
    default_reuse_overhead_ms: float = 1.0
    alpha: float = 0.20
    buckets: Dict[str, BucketStats] = field(default_factory=dict)

    def _stats(self, prefix_tokens: int) -> BucketStats:
        bucket = prefix_length_bucket(prefix_tokens)
        if bucket not in self.buckets:
            self.buckets[bucket] = BucketStats(
                ewma_full_ms_per_token=max(float(self.default_full_ms_per_token), 1e-9),
                ewma_reuse_overhead_ms=max(float(self.default_reuse_overhead_ms), 0.0),
                alpha=float(self.alpha),
            )
        return self.buckets[bucket]

    def update_full(self, token_count: int, latency_ms: float) -> None:
        self._stats(token_count).update_full(token_count, latency_ms)

    def update_reuse(self, matched_prefix_tokens: int, suffix_tokens: int, latency_ms: float, fallback_full_ms_per_token: float) -> None:
        self._stats(matched_prefix_tokens).update_reuse(suffix_tokens, latency_ms, matched_prefix_tokens, fallback_full_ms_per_token)

    def decide(
        self,
        *,
        prefix_tokens: int,
        suffix_tokens: int,
        min_net_saved_ms: float,
        extra_cost_ms: float = 0.0,
        scaffold_discount: float = 1.0,
    ) -> UtilityDecision:
        stats = self._stats(prefix_tokens)
        bucket = prefix_length_bucket(prefix_tokens)
        full_mpt = max(float(stats.ewma_full_ms_per_token or self.default_full_ms_per_token), 1e-9)
        overhead = max(float(stats.ewma_reuse_overhead_ms or self.default_reuse_overhead_ms), 0.0)
        expected_saved = full_mpt * max(int(prefix_tokens), 0)
        expected_cost = overhead * float(scaffold_discount) + max(float(extra_cost_ms), 0.0)
        net = expected_saved - expected_cost
        if net >= float(min_net_saved_ms):
            stats.admitted += 1
            return UtilityDecision(True, bucket, expected_saved, expected_cost, net, "positive_net_utility")
        stats.bypassed_negative += 1
        return UtilityDecision(False, bucket, expected_saved, expected_cost, net, "negative_net_utility")

    def snapshot(self) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {}
        for bucket, stats in sorted(self.buckets.items()):
            prefix = f"utility_bucket_{bucket.replace('-', '_').replace('+', 'plus')}"
            out[f"{prefix}_observations"] = stats.observations
            out[f"{prefix}_full_ms_per_token"] = float(stats.ewma_full_ms_per_token)
            out[f"{prefix}_reuse_overhead_ms"] = float(stats.ewma_reuse_overhead_ms)
            out[f"{prefix}_admitted"] = stats.admitted
            out[f"{prefix}_negative_bypass"] = stats.bypassed_negative
        return out
