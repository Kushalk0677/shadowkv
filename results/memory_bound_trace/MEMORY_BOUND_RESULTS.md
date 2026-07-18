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

Raw per-seed artifacts: `qwen2.5/qwen2.5_14b/`.

| Metric | No cache | Native | MeritKV | Delta |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 58.4 ms | 54.7 ms | 54.6 ms | -6.51% |
| Phase 3 recovery* | 0% | 100.0% | 100.0% | +0.0 pp |
| Victim misses | 15/15 | 5.7/15 | 5.1/15 | -0.6 |
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

Raw per-seed artifacts: `qwen2.5/qwen2.5_32b/`.

| Metric | No cache | Native | MeritKV | Delta |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 72.8 ms | 59.4 ms | 59.6 ms | -18.13% |
| Phase 3 recovery* | 0% | 54.8% | 91.6% | +36.8 pp |
| Victim misses | 15/15 | 11.8/15 | 3.2/15 | -8.6 |
| Declined admissions (Ph1) | - | 0 | 34.6 | - |
| Phase 1 hit rate | 0% | 100.0% | 100.0% | +0.0 pp |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40.0 | 5.4 |
| 1 | Prefixes declined | 0 | 34.6 |
| 2 | Victim misses | 11.8/15 | 3.2/15 |
| 3 | Phase 3 recovery | 54.8% | 91.6% |

### Gemma 4 12B

KV cache: 160 KB/token.  Available for cache: 65 GB.  Pressure (fraction of cache consumed by trace): 0.0052.

Raw per-seed artifacts: `gemma4/gemma_4_12b/`.

| Metric | No cache | Native | MeritKV | Delta |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 45.3 ms | 37.2 ms | 36.9 ms | -18.5% |
| Phase 3 recovery* | 0% | 95.0% | 99.5% | +4.5 pp |
| Victim misses | 15/15 | 8/15 | 3/15 | -5 |
| Declined admissions (Ph1) | - | 0 | 30 | - |
| Phase 1 hit rate | 0% | 100% | 100% | - |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40 | 10 |
| 1 | Prefixes declined | 0 | 30 |
| 2 | Victim misses | 8/15 | 3/15 |
| 3 | Phase 3 recovery | 95.0% | 99.5% |

### Gemma 4 31B

KV cache: 256 KB/token.  Available for cache: 28 GB.  Pressure (fraction of cache consumed by trace): 0.0194.

Raw per-seed artifacts: `gemma4/gemma_4_31b/`.

| Metric | No cache | Native | MeritKV | Delta |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 79.1 ms | 68.4 ms | 66.5 ms | -15.9% |
| Phase 3 recovery* | 0% | 54.8% | 95.0% | +40.2 pp |
| Victim misses | 15/15 | 12/15 | 2.5/15 | -9.5 |
| Declined admissions (Ph1) | - | 0 | 35 | - |
| Phase 1 hit rate | 0% | 100% | 71% | - |

| Phase | Decision | Native | MeritKV |
|-------|----------|:------:|:-------:|
| 1 | Prefixes admitted | 40 | 5 |
| 1 | Prefixes declined | 0 | 35 |
| 2 | Victim misses | 12/15 | 2.5/15 |
| 3 | Phase 3 recovery | 54.8% | 95.0% |

---
## NVIDIA T4

### Phi-3-mini

KV cache: 48 KB/token.  Available for cache: 6.9 GB.  Pressure (fraction of cache consumed by trace): 0.015.

| Metric | No cache | Native | MeritKV | Delta |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 63.2 ms | 59.1 ms | 58.9 ms | -6.80% |
| Phase 3 recovery* | 0% | 100.0% | 100.0% | +0.0 pp |
| Victim misses | 15/15 | 13.2/15 | 13.2/15 | 0 |
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

| Metric | No cache | Native | MeritKV | Delta |
|--------|:--------:|:------:|:-------:|:-----------:|
| Mean latency | 78.4 ms | 62.1 ms | 62.3 ms | -20.54% |
| Phase 3 recovery* | 0% | 2.0% | 97.3% | +95.3 pp |
| Victim misses | 15/15 | 14.8/15 | 0.6/15 | -14.2 |
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
| BWell | Qwen2.5-14B | 0.006 | 12.6/40 | 100.0% | +0 pp | 5.1/15 |
| BWell | Qwen2.5-32B | 0.019 | 5.4/40 | 91.6% | +37 pp | 3.2/15 |
| BWell | Gemma 4 12B | 0.0052 | 10/40 | 99.5% | +4.5 pp | 3/15 |
| BWell | Gemma 4 31B | 0.0194 | 5/40 | 95.0% | +40 pp | 2.5/15 |
| T4 | Phi-3-mini | 0.015 | 30.0/40 | 100.0% | +0 pp | 13.2/15 |
| T4 | Qwen2.5-7B | 1.000 | 12.2/40 | 97.3% | +95 pp | 0.6/15 |

* Recovery rate: fraction of Phase 3 requests that hit cache (hit-rate metric, not latency).

In the metric tables, latency deltas compare MeritKV with no cache; recovery and victim-miss deltas compare MeritKV with native caching.

Qwen2.5 and Gemma 4 Blackwell rows include raw aggregate and per-seed trace JSONs under `qwen2.5/` and `gemma4/`.

