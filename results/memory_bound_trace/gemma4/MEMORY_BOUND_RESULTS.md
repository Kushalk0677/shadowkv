# Memory-Bound Trace Results -- Gemma 4 on Blackwell

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

## Blackwell RTX PRO 6000

### Gemma 4 12B

KV cache: 160 KB/token.  Available for cache: 65 GB.
Pressure (fraction of cache consumed by trace): 0.0052.

| Metric | No cache | Native | MeritKV | D vs native |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 45.3 ms | 37.2 ms | 36.9 ms | -18.54% |
| Phase 3 recovery* | 0% | 95.0% | 99.5% | +4.5 pp |
| Victim misses | 15/15 | 8/15 | 3/15 | -5 |
| Declined admissions (Ph1) | -- | 0 | 30 | -- |
| Phase 1 hit rate | 0% | 100% | 100% | -- |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40 | 10 |
| 1 | Prefixes declined | 0 | 30 |
| 2 | Victim misses | 8/15 | 3/15 |
| 3 | Phase 3 recovery | 95.0% | 99.5% |

### Gemma 4 31B

KV cache: 256 KB/token.  Available for cache: 28 GB.
Pressure (fraction of cache consumed by trace): 0.0194.

| Metric | No cache | Native | MeritKV | D vs native |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 79.1 ms | 68.4 ms | 66.5 ms | -15.93% |
| Phase 3 recovery* | 0% | 54.8% | 95.0% | +40.2 pp |
| Victim misses | 15/15 | 12/15 | 2.5/15 | -9 |
| Declined admissions (Ph1) | -- | 0 | 35 | -- |
| Phase 1 hit rate | 0% | 100% | 71% | -- |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40 | 5 |
| 1 | Prefixes declined | 0 | 35 |
| 2 | Victim misses | 12/15 | 2.5/15 |
| 3 | Phase 3 recovery | 54.8% | 95.0% |

---
## Summary

| GPU | Model | Pressure | Native Recov. | MeritKV Recov. | Delta | Victim Misses (N/M) |
|:---:|-------|:--------:|:-------------:|:--------------:|:-----:|:-------------------:|
| BWell | Gemma 4 12B | 0.0052 | 95.0% | 99.5% | +4.5pp | 8/3 |
| BWell | Gemma 4 31B | 0.0194 | 54.8% | 95.0% | +40.2pp | 12/2.5 |

* Recovery rate: fraction of Phase 3 requests that hit cache (hit-rate metric, not latency).
