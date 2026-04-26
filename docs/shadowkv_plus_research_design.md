# ShadowKV++ Research Design

## Positioning

ShadowKV++ reframes ShadowKV from a cache mechanism into a policy-driven inference controller.

The contribution is not "prefix caching exists." The contribution is a serving-time controller that decides:

1. whether reuse should be attempted,
2. which reuse mode should be attempted,
3. how deep the system should speculate,
4. what logical layer/segment reuse ratio is justified,
5. when the expected waste is higher than the expected benefit.

## Core objective

ShadowKV++ optimizes a net-utility objective:

```text
utility = expected_latency_benefit - expected_reuse_cost - expected_waste
```

This makes waste a first-class control variable rather than a post-hoc diagnostic.

## Components

### 1. AdaptiveReuseController

Location: `src/proactive_kv_cache/controller.py`

Inputs:
- prompt length
- token entropy
- exact prefix match length
- semantic similarity to known prefix families
- shared scaffold hints from templated/RAG workloads
- online EWMA hit-rate and waste feedback
- estimated prefill ms/token and reuse overhead

Outputs:
- `bypass`
- `exact`
- `semantic_partial`
- speculation depth
- reusable prefix token count
- logical layer reuse ratio
- expected benefit/cost/waste
- confidence and explanation reason

### 2. SemanticKVIndex

Location: `src/proactive_kv_cache/semantic.py`

A dependency-free token sketch index. It uses hashed unigram and bigram features to estimate semantic/structural neighbourhoods cheaply in the serving path.

Important correctness rule:
- real backends only perform exact-prefix KV reuse;
- approximate semantic partial reuse is enabled only for `FakeBackend` unless explicitly overridden.

This means the repo can study the opportunity of semantic approximate reuse without silently making unsafe correctness claims.

### 3. Fine-grained reuse accounting

ShadowKV++ records a logical layer reuse ratio per admitted plan.

This is currently a research-control signal and metric. Real tensor-level partial layer KV reuse is intentionally not claimed unless a backend implementation supports it.

### 4. Offline policy learning

Location: `src/proactive_kv_cache/policy_learning.py`

The offline learner parses completed benchmark JSON files and searches conservative deployment thresholds from ShadowKV++ rows only:
- minimum reuse density,
- maximum waste ratio,
- minimum hit rate.

This makes the policy auditable and reproducible from prior experiment logs.

## What counts as evidence

### Smoke/regression evidence
- fake backend runs,
- unit tests,
- synthetic matrix.

### Performance evidence
- Hugging Face or vLLM runs,
- repeated seeds,
- public datasets,
- raw + templated modes,
- no-cache/reactive/strict-reactive/ShadowKV/ShadowKV++ comparisons.

### Publishable ShadowKV++ metrics
Report at least:
- speedup vs no cache,
- speedup vs strict reactive prefix cache,
- waste compute ratio,
- policy net utility,
- policy exact/semantic/bypass counts,
- semantic match count,
- semantic partial hit count,
- layer reuse events,
- cache disable reason where applicable.

## Main paper claim after integration

> ShadowKV++ is a waste-aware, policy-driven KV reuse controller that combines exact prefix reuse, semantic neighbourhood signals, fine-grained reuse accounting, and offline-learned deployment gates. It improves prefix-rich LLM serving workloads while explicitly avoiding or disabling negative-utility reuse regimes.

## Claims not made by this code

The code does not claim that approximate semantic KV reuse is always numerically safe on arbitrary transformer backends. Real backends use exact-prefix reuse unless a backend-specific correctness implementation is added and tested.
