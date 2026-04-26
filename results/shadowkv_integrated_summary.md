# ShadowKV result summary

Parsed runs: 1263

| engine | prompt_mode | n | mean speedup | mean hit rate | mean waste | mean reuse density |
|---|---:|---:|---:|---:|---:|---:|
| frequency_speculative | raw | 109 | 1.015 | 0.000 | 0.781 | 0.142 |
| frequency_speculative | templated | 100 | 1.154 | 0.000 | 0.000 | 0.376 |
| greedy_prefix_cache | raw | 109 | 1.120 | 0.000 | 0.000 | 0.028 |
| greedy_prefix_cache | templated | 100 | 1.162 | 0.000 | 0.000 | 0.405 |
| no_cache | raw | 109 | 1.000 | 0.000 | 0.000 | 0.000 |
| no_cache | templated | 100 | 1.000 | 0.000 | 0.000 | 0.000 |
| reactive_prefix_cache | raw | 109 | 1.135 | 0.000 | 0.000 | 0.034 |
| reactive_prefix_cache | templated | 100 | 1.142 | 0.000 | 0.000 | 0.330 |
| shadow_kv | raw | 109 | 1.117 | 0.000 | 0.671 | 0.075 |
| shadow_kv | templated | 100 | 1.236 | 0.000 | 0.050 | 0.384 |
| shadow_kv_plus | raw | 9 | 1.367 | 0.000 | 0.000 | 0.215 |
| strict_reactive_prefix_cache | raw | 109 | 1.139 | 0.000 | 0.000 | 0.020 |
| strict_reactive_prefix_cache | templated | 100 | 1.174 | 0.000 | 0.000 | 0.384 |

## Learned deployment gate

The dependency-free learner searches conservative thresholds for enabling ShadowKV++ on future workload families.

```json
{
  "min_reuse_density": 0.12,
  "max_waste_ratio": 0.0,
  "min_cache_hit_rate": 0.0,
  "estimated_accuracy": 1.0,
  "n_training_rows": 9
}
```

## Interpretation guidance

- Treat fake-backend runs as regression/smoke evidence only.
- Treat HF/vLLM repeated seeded runs as performance evidence.
- ShadowKV++ metrics to report: policy net utility, semantic match rate, semantic partial hits, layer reuse events, and waste ratio.
