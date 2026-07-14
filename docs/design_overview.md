# System Design Overview

## Architecture

MeritKV comprises five components that work together per-request:

```
Request x
    |
    v
Raw-Mode Gate  --> bypass (no cache overhead)
    | admit
    v
AdaptiveReuseController
    |--- queries TieredStateBank for prefix matches
    |--- queries SemanticKVIndex for paraphrase similarity
    |--- computes U = B - C - W
    |--- returns ReusePlan: strategy, score, reusable prefix length
    v
Backend (HF / SGLang / vLLM)
    |
    v
ReusePlan logged to Offline Learner
```

## TieredStateBank

The bank stores KV entries keyed by prefix token sequences with longest-prefix lookup via a radix trie. Per-prefix statistics:

- **EWMA frequency** f(p) — how often this prefix has been observed recently
- **Observation count** n(p) — total times this prefix has been seen
- **Branching factor** b(p) — how many distinct suffixes follow this prefix
- **Memory footprint** m(p) — VRAM consumed by the cached KV tensor

Entries start in a CPU tier and are promoted to GPU after sufficient reuse hits. Demotion happens under memory pressure using an LRU-like policy.

### Why EWMA for frequency?

The exponential weighted moving average (alpha = 0.20) gives more weight to recent observations than old ones, so the frequency estimate adapts to workload shifts. A simple hit counter would be slow to react when a previously popular prefix stops appearing.

## AdaptiveReuseController

The controller calls Plan(x) per request and returns a ReusePlan with: strategy (bypass/exact/semantic_partial), speculate depth, reusable prefix length, expected benefit/cost/waste (ms), score, confidence, and reason.

### Feature Extraction

Four features feed into every planning decision:

```
ts (template score) = 0.85 for templated/RAG, else 1 if shared hint > 0
ls (length score)   = min(|x|/128, 1.0)
es (entropy score)  = 1 - 0.35 * min(H(x)/8, 1.0)  where H(x) is token entropy
health              = clip(0.65 + 0.35*eh - 0.45*ew, 0.05, 1.0)
```

Health tracks the engine's recent effectiveness: high hit rate (eh) increases it, high waste (ew) decreases it. This is the waste-adaptive feedback loop.

### Exact-Prefix Branch

When the bank returns a prefix match of length k > 0:

```
Be = k * max(beta, 0.05)          -- benefit (ms saved by reuse)
Ce = delta_r + 0.02 * (|x| - k)    -- cost (overhead of cache lookup + suffix compute)
We = Ce * max(ew, 0.02)             -- waste (cost discounted by waste ratio)
Ue = Be - Ce - We                    -- net utility
```

Plan is exact if Ue >= 0, otherwise bypass.

**Why this formula?** Beta (ms/token) and delta_r (ms overhead) are calibrated from hardware measurements (6-point prefill latency profile). The waste discount ensures that when the engine's recent waste ratio is high, it requires a larger benefit to admit a reuse.

### Semantic Branch

When no exact match exists but the semantic index returns a candidate with similarity sigma >= 0.58:

```
Bs = 0.72 * kr * max(beta, 0.05)
Ws = 0.35 * Bs * (1 - min(sigma, 0.98)) + Cs * max(ew, 0.03)
Us = (Bs - Cs - Ws) * health
```

The 0.72 discount on benefit accounts for semantic uncertainty. Admitted semantic_partial plans are executed only when Us >= 0 and the backend's allows_approximate_semantic_reuse flag is True.

### Risk-Averse Extension (Coupled Utility)

The base U = B - C - W treats benefit, cost, and waste as independent. In practice they are coupled through the model's VRAM footprint kappa:

```
Ue(lambda) = Ue - lambda * kappa * Be * max(ew, 0.02)
Us(lambda) = Us - lambda * kappa * Bs * max(ew, 0.02)
```

lambda >= 0 is the risk-aversion parameter (default 0.15). At lambda = 0 the base policy is recovered exactly. This is analogous to mean-variance portfolio optimisation.

**Why a coupling penalty?** High-kappa models (Phi-3: 0.375 MB/token) incur larger waste penalties because each wasted precompute occupies more memory. The covariance between benefit and waste is negative (correlation -0.24), meaning high-benefit admits also correlate with high waste. The penalty automatically blocks the worst semantic-mode failures on Phi-3.

### EWMA Feedback

After each request, two EWMA estimates update:

```
eh <- (1-alpha) * eh + alpha * 1[hit]
ew <- (1-alpha) * ew + alpha * W_obs
```

alpha = 0.20. This implements the waste-adaptive behaviour: sustained waste increases ew, which decreases U for future admits, eventually triggering bypass.

## Raw-Mode Conservative Gate

On raw workloads (no shared prefix, no template), the engine starts in no-store, no-speculate mode. It graduates to active caching only when five conditions hold:

1. >= 8 observations of the candidate prefix
2. Frequency EWMA >= 0.35
3. Candidate prefix length >= 48 tokens
4. >= 12 total requests seen
5. Estimated net utility >= 8 ms

**Why this gate?** Raw workloads have inherently low prefix reuse. Without this gate, every request pays cache lookup overhead for zero benefit. The gate ensures the engine only enables caching when there's statistical evidence of reuse.

## SemanticKVIndex

The index detects paraphrase-equivalent request families with a 128-dim hash-based token sketch. No sentence encoder on the hot path.

- Unigram hash: h(t) = t * 2654435761 mod 2^32
- Bigram hash: h(t, t') = (t * 1315423911 XOR t' * 97531) mod 2^32
- Each token contributes +/-1 to sketch[h_u mod 128] and +/-0.5 to sketch[h_b mod 128]
- Cosine similarity between sketches approximates structural overlap

In semantic prompt mode, prompts carry a semantic_equivalence_key. Queries within the same family receive a similarity boost of 0.35. Unlabelled prompts rely on sketch similarity alone (0.61-0.74 for instruction-tuned models), which is above the 0.58 detection threshold.

## Parameter Calibration

Seven parameters are calibrated from hardware measurements, not guessed:

| Parameter | Meaning | How Calibrated |
|-----------|---------|----------------|
| beta | Marginal prefill cost (ms/token) | 6-point latency profile (8-96 tokens) |
| delta_r | Reuse overhead (ms) | Y-intercept of latency profile |
| kappa | KV footprint (MB/token) | Estimated from model config |
| k* | Breakeven prefix length | Eq: k* = delta_r / (beta - kappa * 1000 / Bw) |
| k_min | Minimum reusable prefix | 16, safe for 4/5 models (Phi-3 uses breakeven guard) |
| n_min | Minimum prefix observations | 8, derived from EWMA standard error |
| U_min | Minimum utility threshold | beta * k_min - delta_r - memory_penalty - noise_floor |

## Offline Policy Learner

After benchmark runs, the learner grid-searches threshold triples (rho_min, w_max, h_min) predicting (speedup > 1.02) AND (W < 0.35) from MeritKV rows only. Output is a JSON artifact loadable as controller defaults.
