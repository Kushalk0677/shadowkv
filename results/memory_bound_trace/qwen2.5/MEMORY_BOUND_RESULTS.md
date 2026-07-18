# Memory-Bound Trace Results -- Qwen2.5 on Blackwell

3-phase interleaved trace exceeding KV cache capacity midway, forcing evictions.

| Phase | Requests | Content | Purpose |
|-------|:--------:|---------|---------|
| 1 - Fill | 40 | Shared prefix (220 tok) + unique suffix (50 tok) | Fill cache with reusable entries |
| 2 - Churn | 30 | 15 evictors + 15 victims of Phase 1 prefix | Force eviction, measure victim misses |
| 3 - Recovery | 30 | Repeat Phase 1 template | Measure cache recovery after churn |

| Baseline | Description |
|----------|-------------|
| **No cache** | Disabled |
| **Native** | vLLM APC. Admits unconditionally. |
| **MeritKV** | Full U = B - C - W admission gate. |

---

## Qwen2.5-14B

KV cache: 192 KB/token.  Available for cache: 64.4 GB.
Pressure (fraction of cache consumed by trace): 0.006.

| Metric | No cache | Native | MeritKV | D vs native |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 58.4 ms | 54.7 ms | 54.6 ms | -6.51% |
| Phase 3 recovery* | 0% | 100.0% | 100.0% | +0.0 pp |
| Victim misses | 15/15 | 5.7/15 | 5.1/15 | -1 |
| Declined admissions (Ph1) | -- | 0 | 27.4 | -- |
| Phase 1 hit rate | 0% | 100% | 100% | -- |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40 | 12.6 |
| 1 | Prefixes declined | 0 | 27.4 |
| 2 | Victim misses | 5.7/15 | 5.1/15 |
| 3 | Phase 3 recovery | 100.0% | 100.0% |

## Qwen2.5-32B

KV cache: 256 KB/token.  Available for cache: 28.0 GB.
Pressure (fraction of cache consumed by trace): 0.019.

| Metric | No cache | Native | MeritKV | D vs native |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 72.8 ms | 59.4 ms | 59.6 ms | -18.13% |
| Phase 3 recovery* | 0% | 54.8% | 91.6% | +36.8 pp |
| Victim misses | 15/15 | 11.8/15 | 3.2/15 | -8 |
| Declined admissions (Ph1) | -- | 0 | 34.6 | -- |
| Phase 1 hit rate | 0% | 100% | 100% | -- |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40 | 5.4 |
| 1 | Prefixes declined | 0 | 34.6 |
| 2 | Victim misses | 11.8/15 | 3.2/15 |
| 3 | Phase 3 recovery | 54.8% | 91.6% |

---
## Summary

| GPU | Model | Pressure | Native Recov. | MeritKV Recov. | Delta | Victim Misses (N/M) |
|:---:|-------|:--------:|:-------------:|:--------------:|:-----:|:-------------------:|
| BWell | Qwen2.5-14B | 0.006 | 100% | 100% | +0pp | 5.7/5.1 |
| BWell | Qwen2.5-32B | 0.019 | 55% | 92% | +37pp | 11.8/3.2 |

* Recovery rate: fraction of Phase 3 requests that hit cache (hit-rate metric, not latency).
