# Memory-Bound Trace Results

3-phase interleaved trace exceeding KV cache capacity midway, forcing evictions.

| Phase | Requests | Content | Purpose |
|-------|:--------:|---------|---------|
| 1 - Fill | 40 | Shared prefix (220 tok) + unique suffix (50 tok) | Fill cache with reusable entries |
| 2 - Churn | 30 | 15 evictors + 15 victims of Phase 1 prefix | Force eviction, measure victim misses |
| 3 - Recovery | 30 | Repeat Phase 1 template | Measure cache recovery after churn |

| Baseline | Description |
|----------|-------------|
| **No cache** | Disabled |
| **Native** | vLLM APC (Blackwell) / HF prefix cache (T4). Admits unconditionally. |
| **MeritKV** | Full U = B - C - W admission gate. |

---
## Blackwell RTX PRO 6000

### Qwen2.5-14B

KV cache: 192 KB/token.  Available for cache: 64.4 GB.  Pressure (fraction of cache consumed by trace): 0.006.

| Metric | No cache | Native | MeritKV | D vs native |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 58.4 ms | 54.7 ms | 54.6 ms | -6.51% |
| Phase 3 recovery* | 0% | 100.0% | 100.0% | +-0.0 pp |
| Victim misses | 15/15 | 5.7/15 | 5.1/15 | -4% |
| Declined admissions (Ph1) | - | 0 | 27.4 | - |
| Phase 1 hit rate | 0% | 100.0% | 100.0% | +0.0 pp |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40.0 | 12.6 |
| 1 | Prefixes declined | 0 | 27.4 |
| 2 | Victim misses | 5.7/15 | 5.1/15 |
| 3 | Phase 3 recovery | 100.0% | 100.0% |

### Qwen2.5-32B

KV cache: 256 KB/token.  Available for cache: 28.0 GB.  Pressure (fraction of cache consumed by trace): 0.019.

| Metric | No cache | Native | MeritKV | D vs native |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 72.8 ms | 59.4 ms | 59.6 ms | -18.13% |
| Phase 3 recovery* | 0% | 54.8% | 91.6% | +36.8 pp |
| Victim misses | 15/15 | 11.8/15 | 3.2/15 | -57% |
| Declined admissions (Ph1) | - | 0 | 34.6 | - |
| Phase 1 hit rate | 0% | 100.0% | 100.0% | +0.0 pp |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40.0 | 5.4 |
| 1 | Prefixes declined | 0 | 34.6 |
| 2 | Victim misses | 11.8/15 | 3.2/15 |
| 3 | Phase 3 recovery | 54.8% | 91.6% |

---
## NVIDIA T4

### Phi-3-mini

KV cache: 48 KB/token.  Available for cache: 6.9 GB.  Pressure (fraction of cache consumed by trace): 0.015.

| Metric | No cache | Native | MeritKV | D vs native |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 63.2 ms | 59.1 ms | 58.9 ms | -6.80% |
| Phase 3 recovery* | 0% | 100.0% | 100.0% | +0.0 pp |
| Victim misses | 15/15 | 13.2/15 | 13.2/15 | -0% |
| Declined admissions (Ph1) | - | 0 | 10.0 | - |
| Phase 1 hit rate | 0% | 100.0% | 100.0% | +0.0 pp |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40.0 | 30.0 |
| 1 | Prefixes declined | 0 | 10.0 |
| 2 | Victim misses | 13.2/15 | 13.2/15 |
| 3 | Phase 3 recovery | 100.0% | 100.0% |

### Qwen2.5-7B

KV cache: 56 KB/token.  Available for cache: 0.1 GB.  Pressure (fraction of cache consumed by trace): 1.000.

| Metric | No cache | Native | MeritKV | D vs native |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 78.4 ms | 62.1 ms | 62.3 ms | -20.54% |
| Phase 3 recovery* | 0% | 2.0% | 97.3% | +95.3 pp |
| Victim misses | 15/15 | 14.8/15 | 0.6/15 | -95% |
| Declined admissions (Ph1) | - | 0 | 27.8 | - |
| Phase 1 hit rate | 0% | 99.5% | 71.2% | -28.3 pp |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40.0 | 12.2 |
| 1 | Prefixes declined | 0 | 27.8 |
| 2 | Victim misses | 14.8/15 | 0.6/15 |
| 3 | Phase 3 recovery | 2.0% | 97.3% |

---
## Summary

| GPU | Model | Pressure | Admitted | Phase 3 recovery | vs native | Victim misses |
|:---:|-------|:--------:|:--------:|:----------------:|:---------:|:-------------:|
| BWell | Qwen2.5-14B | 0.006 | 12.6/40 | 100.0% | +-0 pp | 5.1/15 |
| BWell | Qwen2.5-32B | 0.019 | 5.4/40 | 91.6% | +37 pp | 3.2/15 |
| T4 | Phi-3-mini | 0.015 | 30.0/40 | 100.0% | +0 pp | 13.2/15 |
| T4 | Qwen2.5-7B | 1.000 | 12.2/40 | 97.3% | +95 pp | 0.6/15 |

* Recovery rate: fraction of Phase 3 requests that hit cache (hit-rate metric, not latency).

