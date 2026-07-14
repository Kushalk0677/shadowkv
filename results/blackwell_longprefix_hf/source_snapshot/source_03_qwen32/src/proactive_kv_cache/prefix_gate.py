from __future__ import annotations

import math

from .config_loader import CONFIG, RuntimeConfig


def transfer_ms_per_token(config: RuntimeConfig = CONFIG) -> float:
    kv_mb = float(config.get('hardware.kv_mb_per_token', 0.375))
    bandwidth_gbps = max(float(config.get('hardware.memory_bandwidth_gbps', 320.0)), 1e-9)
    # GB/s -> MB/ms is numerically GB/s * 1024 / 1000.
    return float(kv_mb / (bandwidth_gbps * 1024.0) * 1000.0)


def breakeven_prefix_len(config: RuntimeConfig = CONFIG) -> int:
    beta = max(float(config.get('hardware.beta_prefill_ms_per_token', 0.60)), 0.0)
    delta = max(float(config.get('hardware.reuse_fixed_overhead_ms', 2.0)), 0.0)
    min_prefix = max(int(config.get('hardware.min_prefix_len', 8)), 1)
    marginal_saving = beta - transfer_ms_per_token(config)
    if marginal_saving <= 1e-9:
        return min_prefix
    return max(int(math.ceil(delta / marginal_saving)), min_prefix)
