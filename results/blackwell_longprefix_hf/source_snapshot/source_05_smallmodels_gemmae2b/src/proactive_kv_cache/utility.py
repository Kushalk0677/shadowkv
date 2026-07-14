from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .base_policy import ReuseStrategy
from .config_loader import CONFIG, RuntimeConfig
from .prefix_gate import breakeven_prefix_len
from .semantic import token_entropy


@dataclass
class AdmissionEvent:
    tokens: Tuple[int, ...]
    exact_match_len: int
    semantic_similarity: float
    semantic_prefix_len: int
    shared_prefix_hint: int | None
    full_ms_per_token: float
    reuse_overhead_ms: float
    ewma_hit_rate: float
    ewma_waste_ratio: float
    max_layer_reuse_ratio: float
    min_utility_ms: float
    semantic_threshold: float
    metadata: Dict | None = None
    semantic_divergence: float | None = None
    tier: str | None = None
    memory_bytes: int = 0
    min_bootstrap_admissions: int = 3
    observation_count: int = 0


@dataclass
class UtilityBreakdown:
    strategy: ReuseStrategy
    reusable_prefix_tokens: int
    speculate_depth_tokens: int
    layer_reuse_ratio: float
    benefit_ms: float
    cost_ms: float
    waste_ms: float
    net_utility_ms: float
    confidence: float
    health: float
    reason: str


class UtilityModel:
    def __init__(self, config: RuntimeConfig = CONFIG) -> None:
        self.config = config

    def _health(self, event: AdmissionEvent) -> float:
        base = float(self.config.get('policy.health.base', 0.65))
        hit_weight = float(self.config.get('policy.health.hit_rate_weight', 0.35))
        waste_weight = float(self.config.get('policy.health.waste_ratio_weight', 0.45))
        low = float(self.config.get('policy.health.min', 0.05))
        high = float(self.config.get('policy.health.max', 1.0))
        value = base + hit_weight * event.ewma_hit_rate - waste_weight * event.ewma_waste_ratio
        return max(min(value, high), low)

    def admission(self, event: AdmissionEvent) -> UtilityBreakdown:
        tokens = event.tokens
        n = max(len(tokens), 1)
        
        # Dynamic hardware profiling for Transfer Cost
        pcie_bw_gbps = float(self.config.get('hardware.pcie_bandwidth_gbps', 15.0))
        pcie_bytes_per_ms = (pcie_bw_gbps * 1024 ** 3) / 1000.0
        transfer_cost_ms = (event.memory_bytes / pcie_bytes_per_ms) if event.tier == 'cpu' else 0.0
        
        # lambda_M: Memory Opportunity Cost
        lambda_m_weight = float(self.config.get('policy.utility.lambda_m_weight', 0.02))
        memory_cost_ms = (event.memory_bytes / (1024 ** 2)) * lambda_m_weight
        
        metadata = event.metadata or {}
        prompt_mode = str(metadata.get('prompt_mode', '')).lower()
        entropy = token_entropy(tokens)
        template_signal = 1.0 if event.shared_prefix_hint else 0.0
        if prompt_mode in {'templated', 'rag'}:
            template_signal = max(template_signal, float(self.config.get('policy.signals.templated_signal', 0.85)))
        length_signal = min(n / float(self.config.get('policy.signals.length_normalizer_tokens', 128.0)), 1.0)
        entropy_signal = 1.0 - min(entropy / float(self.config.get('policy.signals.entropy_normalizer_bits', 8.0)), 1.0) * float(self.config.get('policy.signals.entropy_penalty_weight', 0.35))
        health = self._health(event)
        min_viable_prefix = breakeven_prefix_len(self.config)

        if event.exact_match_len > 0:
            reusable = int(event.exact_match_len)
            if reusable < min_viable_prefix:
                return UtilityBreakdown(
                    strategy='bypass',
                    reusable_prefix_tokens=reusable,
                    speculate_depth_tokens=0,
                    layer_reuse_ratio=0.0,
                    benefit_ms=0.0,
                    cost_ms=event.reuse_overhead_ms,
                    waste_ms=0.0,
                    net_utility_ms=-event.reuse_overhead_ms,
                    confidence=0.0,
                    health=health,
                    reason='exact_prefix_below_breakeven',
                )
            gross = reusable * max(event.full_ms_per_token, 0.05)
            suffix = max(n - reusable, 0)
            cost = event.reuse_overhead_ms + transfer_cost_ms + float(self.config.get('policy.utility.suffix_cost_ms_per_token', 0.02)) * suffix
            waste = cost * max(event.ewma_waste_ratio, 0.02)
            
            # Old score (linear, uncoupled)
            score_old = gross - cost - waste - memory_cost_ms
            
            # Coupled utility: add risk-aversion penalty for memory-heavy admits
            # Where benefit and waste are coupled through model VRAM footprint
            lambda_risk = float(self.config.get('policy.utility.risk_aversion', 0.15))
            vram_factor_mb = event.memory_bytes / (1024 ** 2) if event.memory_bytes > 0 else 0.0
            coupling_penalty = lambda_risk * gross * vram_factor_mb * max(event.ewma_waste_ratio, 0.02)
            
            # U = B - C - W - lambda*M - Risk(B,W)
            score = gross - cost - waste - memory_cost_ms - coupling_penalty
            conf_base = float(self.config.get('policy.utility.exact_confidence_base', 0.45))
            conf_reuse = float(self.config.get('policy.utility.exact_confidence_reuse_weight', 0.35))
            conf_template = float(self.config.get('policy.utility.exact_confidence_template_weight', 0.20))
            confidence = max(
                float(self.config.get('policy.utility.exact_min_confidence', 0.50)),
                min(float(self.config.get('policy.utility.exact_max_confidence', 0.98)), conf_base + conf_reuse * (reusable / n) + conf_template * template_signal),
            ) * health
            layer_ratio = min(
                event.max_layer_reuse_ratio,
                float(self.config.get('policy.utility.exact_layer_base', 0.20)) + float(self.config.get('policy.utility.exact_layer_confidence_weight', 0.30)) * confidence,
            )
            admitted = score >= event.min_utility_ms
            
            if not admitted and event.observation_count < event.min_bootstrap_admissions:
                admitted = True
                reason = 'exact_prefix_bootstrap'
            else:
                reason = 'exact_prefix_net_positive' if admitted else 'exact_prefix_net_negative'

            return UtilityBreakdown(
                strategy='exact' if admitted else 'bypass',
                reusable_prefix_tokens=reusable,
                speculate_depth_tokens=reusable,
                layer_reuse_ratio=layer_ratio,
                benefit_ms=gross,
                cost_ms=cost,
                waste_ms=waste,
                net_utility_ms=score,
                confidence=confidence,
                health=health,
                reason=reason,
            )

        semantic_gate = event.semantic_similarity >= event.semantic_threshold and event.semantic_prefix_len > 0
        if semantic_gate:
            semantic_mode = prompt_mode == 'semantic'
            reuse_fraction = float(self.config.get('policy.utility.semantic_prompt_reuse_fraction' if semantic_mode else 'policy.utility.semantic_default_reuse_fraction', 0.70 if semantic_mode else 0.55))
            reusable = min(event.semantic_prefix_len, max(event.shared_prefix_hint or 0, int(n * reuse_fraction)))
            if reusable < min_viable_prefix:
                return UtilityBreakdown(
                    strategy='bypass',
                    reusable_prefix_tokens=int(reusable),
                    speculate_depth_tokens=0,
                    layer_reuse_ratio=0.0,
                    benefit_ms=0.0,
                    cost_ms=event.reuse_overhead_ms,
                    waste_ms=0.0,
                    net_utility_ms=-event.reuse_overhead_ms,
                    confidence=0.0,
                    health=health,
                    reason='semantic_prefix_below_breakeven',
                )
            benefit_discount = float(self.config.get('policy.utility.semantic_prompt_benefit_discount' if semantic_mode else 'policy.utility.semantic_default_benefit_discount', 0.72 if semantic_mode else 0.55))
            gross = reusable * max(event.full_ms_per_token, 0.05) * benefit_discount
            suffix_cost = float(self.config.get('policy.utility.semantic_prompt_suffix_cost_ms_per_token' if semantic_mode else 'policy.utility.semantic_default_suffix_cost_ms_per_token', 0.04 if semantic_mode else 0.08))
            cost = event.reuse_overhead_ms + transfer_cost_ms + suffix_cost * max(n - reusable, 0)
            uncertainty = max(float(self.config.get('policy.utility.semantic_uncertainty_floor', 0.02)), 1.0 - min(event.semantic_similarity, 0.98))
            uncertainty_weight = float(self.config.get('policy.utility.semantic_prompt_uncertainty_weight' if semantic_mode else 'policy.utility.semantic_default_uncertainty_weight', 0.35 if semantic_mode else 1.0))
            waste_floor = float(self.config.get('policy.utility.semantic_prompt_waste_floor' if semantic_mode else 'policy.utility.semantic_default_waste_floor', 0.03 if semantic_mode else 0.05))
            waste = gross * uncertainty * uncertainty_weight + cost * max(event.ewma_waste_ratio, waste_floor)
            
            # gamma_Q: Semantic Divergence penalty
            gamma_q = 0.0
            if event.semantic_divergence is not None:
                gamma_q_weight = float(self.config.get('policy.utility.gamma_q_weight', 1.0))
                gamma_q = gross * max(float(event.semantic_divergence), 0.0) * gamma_q_weight
                
            # U = B - C - W - lambda*M - gamma*Q - coupling_penalty
            lambda_risk = float(self.config.get('policy.utility.risk_aversion', 0.15))
            vram_mb = event.memory_bytes / (1024 ** 2) if event.memory_bytes > 0 else 0.0
            coupling = lambda_risk * gross * vram_mb * max(event.ewma_waste_ratio, 0.02)
            score = (gross - cost - waste - memory_cost_ms - gamma_q - coupling) * health
            confidence_base = float(self.config.get('policy.utility.semantic_prompt_confidence_base' if semantic_mode else 'policy.utility.semantic_default_confidence_base', 0.70 if semantic_mode else 0.55))
            confidence_cap = float(self.config.get('policy.utility.semantic_prompt_confidence_cap' if semantic_mode else 'policy.utility.semantic_default_confidence_cap', 0.94 if semantic_mode else 0.90))
            confidence = min(
                confidence_cap,
                event.semantic_similarity * (
                    confidence_base
                    + float(self.config.get('policy.utility.semantic_confidence_template_weight', 0.20)) * template_signal
                    + float(self.config.get('policy.utility.semantic_confidence_length_weight', 0.10)) * length_signal
                ),
            ) * entropy_signal * health
            layer_weight = float(self.config.get('policy.utility.semantic_prompt_layer_weight' if semantic_mode else 'policy.utility.semantic_default_layer_weight', 0.55 if semantic_mode else 0.45))
            layer_ratio = min(event.max_layer_reuse_ratio, max(float(self.config.get('policy.utility.semantic_min_layer_ratio', 0.10)), layer_weight * confidence))
            admitted = score >= event.min_utility_ms
            return UtilityBreakdown(
                strategy='semantic_partial' if admitted else 'bypass',
                reusable_prefix_tokens=int(reusable),
                speculate_depth_tokens=int(reusable),
                layer_reuse_ratio=layer_ratio,
                benefit_ms=gross,
                cost_ms=cost,
                waste_ms=waste,
                net_utility_ms=score,
                confidence=confidence,
                health=health,
                reason='semantic_partial_net_positive' if admitted else 'semantic_partial_net_negative',
            )

        return UtilityBreakdown(
            strategy='bypass',
            reusable_prefix_tokens=0,
            speculate_depth_tokens=0,
            layer_reuse_ratio=0.0,
            benefit_ms=0.0,
            cost_ms=0.0,
            waste_ms=0.0,
            net_utility_ms=-event.reuse_overhead_ms,
            confidence=0.0,
            health=health,
            reason='no_reuse_signal',
        )
