# Learned Admission Baseline Math

Raw artifacts keep stable engine IDs: `shadow_kv_plus` displays as MeritKV, `shadow_kv` displays as MeritKV-Sem, and `shadow_kv_plus_lite` displays as MeritKV-Lite.


This note documents the corrected learned admission baseline used to compare against MeritKV's hand-derived per-request utility rule. The baseline is intentionally small: it does not learn a new cache system, a new semantic matcher, or a new KV execution path. It only learns the final admit/bypass decision from logged request-level features.

## High-Level Idea

For each request, MeritKV discovers candidate exact-prefix and semantic-prefix reuse opportunities. The learned baseline sees decision-time features for that same request and predicts whether admitting reuse is likely to be beneficial for this request.

The learned model answers:

```text
Given this request and this candidate cache match, should reuse be admitted?
```

The corrected baseline is not trained to imitate MeritKV's admit/bypass decision. It is trained from observed outcomes of admitted reuse. Ordinary bypasses are missing counterfactuals, so the training harness uses a small epsilon-greedy exploration rate during Phase 1 to force some bypass-eligible candidates through the reuse path and collect outcome labels.

## Outcome Labels

For an admitted request, the trace records whether the attempted reuse was actually beneficial. The label is:

```text
y = 1  if admitted reuse produced positive estimated net savings
y = 0  if admitted reuse failed or produced non-positive estimated net savings
```

For a successful reuse hit, the approximate realized net saving is:

```math
S = \widehat{c}_{\mathrm{full}} \cdot k - \max(L_{\mathrm{reuse}} - \widehat{c}_{\mathrm{full}} \cdot n_{\mathrm{suffix}}, 0)
```

where:

```text
k                    = reused prefix length
n_suffix             = recomputed suffix token count
L_reuse              = observed latency of the admitted reuse request
c_full_hat           = pre-decision EWMA full-prefill milliseconds per token
```

The label is then:

```math
y = \mathbb{1}[S > 0]
```

If the policy attempted admission but the cache path failed or could not reuse, the trace records:

```math
y = 0
```

Bypassed requests without exploration are not used as training labels, because their counterfactual outcome is unknown.

## Exploration For Counterfactual Coverage

During the training phase only, the engine can force a small fraction of bypass-eligible candidates to execute reuse anyway:

```math
\Pr[\mathrm{force\ admit}] = \epsilon
```

The default is:

```math
\epsilon = 0.10
```

The exploration gate is deterministic from the request index, request tokens, and seed, so runs are reproducible while repeated templated prompts still receive independent exploration draws. Exploration is disabled during held-out evaluation.

This fixes the main methodological issue with an imitation baseline: the learned model is trained from realized admit outcomes, not from MeritKV's own binary decision.

## Two Learned Variants

The experiment trains two logistic-regression policies from the same observed outcome labels.

### Learned-Raw

This variant uses request/cache state features only:

```text
matched_prefix_len
exact_match_len
semantic_prefix_len
semantic_lcp_len
semantic_similarity
semantic_match_available
token_count
prefix_ratio
shared_prefix_hint_tokens
ewma_hit_rate
ewma_waste_ratio
is_templated
is_semantic
is_rag
is_raw
```

This asks whether a simple learned model can discover a useful admission boundary from raw decision-time signals without seeing MeritKV's own utility components.

### Learned-UtilityComponents

This variant uses the raw features plus MeritKV's B/C/W-style intermediate quantities:

```text
policy_expected_benefit_ms
policy_expected_cost_ms
policy_expected_waste_ms
policy_confidence
policy_health
```

This asks whether a data-driven reweighting of MeritKV's utility components can beat the fixed hand-derived rule.

The two variants answer different questions and should be reported separately.

## Standardization

For feature vector `x(r)`, each feature is standardized using training-set statistics:

```math
\tilde{x}_i = \frac{x_i - \mu_i}{\sigma_i}
```

where:

```math
\mu_i = \text{mean of feature } i \text{ on the training traces}
```

and:

```math
\sigma_i = \text{standard deviation of feature } i \text{ on the training traces}
```

If a feature has zero scale, the implementation uses scale `1.0` to avoid division by zero.

## Logistic Admission Model

The learned policy computes:

```math
z = b + \sum_{i=1}^{d} w_i \tilde{x}_i
```

and converts it to an admission probability:

```math
p_{\mathrm{admit}} = \sigma(z) = \frac{1}{1 + e^{-z}}
```

The decision threshold is:

```math
\hat{y} = \mathbb{1}[p_{\mathrm{admit}} \ge \tau]
```

where `tau` is selected on a deterministic validation split from:

```math
\tau \in \{0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80\}
```

Older policy JSON files without a saved threshold fall back to:

```math
\tau = 0.50
```

## Training Objective

The baseline uses L2-regularized logistic regression:

```math
\mathcal{L}(w,b) = -\sum_{j=1}^{n} \alpha_{y_j}\left[y_j \log p_j + (1-y_j)\log(1-p_j)\right] + \lambda \lVert w \rVert_2^2
```

where:

```math
p_j = \sigma\left(b + \sum_i w_i \tilde{x}_{j,i}\right)
```

`alpha_y` is the class-balancing weight. The implementation uses:

```text
sklearn.linear_model.LogisticRegression
C in {0.1, 1.0, 10.0}, selected on validation
class_weight = "balanced"
max_iter = 2000
```

`C` is inverse regularization strength in scikit-learn's convention. The saved policy JSON records the selected `C`, the selected decision threshold, and validation metadata when a validation split is possible.

## Runtime Policy

At runtime, the learned model does not create a new cache candidate. It only replaces the final admit/bypass gate after candidate discovery.

```text
if p_admit < tau:
    bypass
else:
    admit if an executable exact match exists
    otherwise admit guarded semantic reuse only if the backend/mode allows it
    otherwise bypass
```

In equation form:

```math
\pi_{\mathrm{learned}}(r) =
\begin{cases}
\text{bypass}, & p_{\mathrm{admit}}(r) < \tau \\
\text{exact reuse}, & p_{\mathrm{admit}}(r) \ge \tau \land k_e > 0 \\
\text{guarded semantic reuse}, & p_{\mathrm{admit}}(r) \ge \tau \land k_s > 0 \land \text{semantic guard passes} \\
\text{bypass}, & \text{otherwise.}
\end{cases}
```

The learned model is not allowed to blindly execute semantic reuse. It can only admit reuse through a path the engine already considers executable.

## Saved Model Files

The corrected driver writes two policy files:

```text
results/learned_baseline/learned_policy_raw.json
results/learned_baseline/learned_policy_utility.json
```

It also writes a compatibility alias for the utility variant:

```text
results/learned_baseline/learned_policy.json
```

Each JSON stores:

```text
model_type
label_mode
feature_variant
feature_names
coef
intercept
mean
scale
train_accuracy
train_samples
train_positive
train_negative
decision_threshold
selected_C
validation_samples
validation_balanced_accuracy
note
```

Inference only needs the JSON coefficients, means/scales, feature names, and threshold.

## Evaluation

Held-out Phase 3 runs both learned policies as real engines and compares them against:

```text
no_cache
shadow_kv              # MeritKV-Sem
shadow_kv_plus_lite    # MeritKV-Lite
shadow_kv_plus         # MeritKV
```

The combined summary is:

```text
results/learned_baseline/phase3_summary.json
```

Variant-specific summaries are also written:

```text
results/learned_baseline/phase3_summary_raw.json
results/learned_baseline/phase3_summary_utility.json
```

MeritKV-Lite (`shadow_kv_plus_lite`) is the in-package capacity/break-even-style comparator. The important metrics are downstream system metrics, not training accuracy:

```text
mean_speedup
mean_waste
mean_hit_rate
learned_policy_admit_total
learned_policy_bypass_total
learned_policy_flip_to_admit_total
learned_policy_flip_to_bypass_total
```

## Interpretation

If Learned-Raw performs well, then much of the admission decision can be recovered from raw request/cache signals.

If Learned-UtilityComponents improves over MeritKV, then B/C/W are useful but may benefit from learned reweighting.

If MeritKV beats both learned variants, that supports the claim that the hand-derived per-request utility rule is doing real work and is not trivially replaced by a small generic classifier.

If either learned variant wins, that is also a useful result: it identifies where a future learned admission controller can improve the fixed formula.

