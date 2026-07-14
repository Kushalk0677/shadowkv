# Engine Failure Analysis

## Overview

Across 750 benchmark runs, the overall failure rate (speedup < 1.0) is 17/750 = 2.3%. All failures involve Qwen2.5-1.5B (14 runs) or Phi-3 (3 runs). This document explains the root cause of each failure mode and how the engine addresses it.

## Phi-3 Raw-Mode Failure (Speedup 0.895x)

### The Symptom

The worst single run is Phi-3 on CNN/DailyMail in raw mode: speedup 0.895x, waste 1.000. This means the engine was slower than no-cache, and every speculative precompute was wasted.

### Root Cause

Phi-3 has a KV footprint of kappa = 0.375 MB/token — 14-21x larger than the other four models. This means each cached KV entry consumes proportionally more VRAM and takes longer to transfer.

The engine admitted a 48-token speculative precompute because Phi-3's BPE tokeniser compressed the long article into fewer tokens with artificially low observed entropy. The breakeven analysis (Eq. k*) shows that on a T4 GPU (320 GB/s bandwidth), Phi-3 needs k* = 449 tokens of prefix reuse just to break even. The 48-token precompute yields net utility of only 1.25 ms before waste discounting. Any cache miss makes U < 0.

### The Fix: Memory-Breakeven Guard

A guard was added to the speculation policy: when a model's kappa exceeds 0.05 MB/token, the engine rejects precompute candidates shorter than the analytic breakeven k* (449 tokens for Phi-3 on T4). This is a **hard constraint** derived from hardware physics — it prevents the failure by ensuring no speculative precompute is admitted for prefixes that cannot break even under any reuse scenario.

The failure does not recur in seeds 789 or 999, consistent with a specific unlucky request ordering in seed 42. The guard prevents it from recurring regardless of ordering.

## Qwen2.5 Semantic-Mode Failures (14 runs)

### The Symptom

Seven of the ten worst runs involve Qwen2.5-1.5B in semantic or templated mode with speedup < 1.0 and zero waste. The zero waste means the engine attempted reuse but the matched prefix was too short for the benefit to cover the overhead.

### Root Cause

Qwen2.5's tokeniser produces short matched prefixes (typically 8-12 tokens) on many datasets. When the matched prefix is short, the estimated benefit Be = beta * k is small, while the fixed overhead delta_r is the same regardless of prefix length. On T4, delta_r = 8.21 ms for Qwen2.5. A 10-token match gives Be = 0.741 * 10 = 7.41 ms, which is less than 8.21 + suffix cost, making U negative.

### Why It's Not Fixed by the Breakeven Guard

The breakeven guard only applies to **speculative precomputes** (storing before a request arrives). Qwen2.5's failures are **reactive** — the engine finds a short cached prefix and attempts to use it, but the benefit doesn't cover the overhead. This is a fundamental property of Qwen2.5's tokeniser producing short overlapping prefixes across diverse prompts.

### Where the Coupling Penalty Helps

The risk-averse extension adds lambda * kappa * B * max(ew, 0.02) to the penalty. For Qwen2.5 (kappa = 0.027), the effect is modest (coupling ratio = 0.39x benefit), blocking 32 admits across all runs but not eliminating the failure entirely. This is expected: short-prefix failures are a tokeniser issue, not a VRAM coupling issue.

## Phi-3 Semantic-Mode Failures (2 runs)

### The Symptom

Phi-3 has two semantic-mode failures in the 750-run set (not visible in the top-10 worst runs table, which only shows speedups down to 0.962x).

### Root Cause

Same mechanism as the raw-mode failure: Phi-3's large kappa means every cache miss is expensive. In semantic mode, the engine attempts reuse based on semantic similarity rather than exact prefix match, which increases the chance of short or mismatched prefixes.

### The Fix: Coupling Penalty

The risk-averse extension blocks 117 of 178 Phi-3 semantic-mode admits (coupling ratio = 4.75x benefit). At lambda = 0.15, all Phi-3 failures (raw and semantic) are eliminated. The raw-mode failure is handled by the breakeven guard; the semantic-mode failures are handled by the coupling penalty.

## Engine-Level Statistics

| Metric | Value |
|--------|-------|
| Total benchmark runs | 750 |
| Total failures (speedup < 1.0) | 17 (2.3%) |
| Qwen2.5-1.5B failures | 14 |
| Phi-3 failures | 3 |
| Semantic opportunities detected | 691 |
| Admitted semantic plans | 691 (at U >= 0) |
| Cumulative net utility of admitted plans | 227,000 ms |

## Summary: Two Mechanisms, Two Failure Modes

| Failure | Model | Cause | Fix | Type |
|---------|-------|-------|-----|------|
| Raw-mode speedup 0.895x | Phi-3 | 48-token precompute on k* = 449 model | Breakeven guard | Hard constraint |
| Semantic speedup < 1.0 | Phi-3 | High kappa makes every miss expensive | Coupling penalty | Soft preference |
| Semantic speedup < 1.0 | Qwen2.5 | Short matched prefixes from tokeniser | Not fully fixable; coupling penalty helps modestly | Architectural limit |

The breakeven guard and coupling penalty are **complementary**: the guard prevents structurally impossible reuse, while the penalty discourages high-risk reuse. Both are needed for different failure modes.
