from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal, Tuple


ReuseStrategy = Literal['bypass', 'exact', 'semantic_partial']


@dataclass
class ReusePlan:
    strategy: ReuseStrategy
    speculate_depth_tokens: int
    reusable_prefix_tokens: int
    layer_reuse_ratio: float
    expected_benefit_ms: float
    expected_cost_ms: float
    expected_waste_ms: float
    score: float
    confidence: float
    reason: str


class PolicyController(ABC):
    @abstractmethod
    def update_feedback(self, *, hit: bool, wasted_ratio: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def plan(
        self,
        *,
        tokens: Tuple[int, ...],
        exact_match_len: int,
        semantic_similarity: float,
        semantic_prefix_len: int,
        shared_prefix_hint: int | None,
        full_ms_per_token: float,
        reuse_overhead_ms: float,
        metadata: Dict | None = None,
    ) -> ReusePlan:
        raise NotImplementedError
