## oracle_future_reuse

Target meaning:
- bespoke offline oracle upper bound for evaluation
- has access to the full future request trace before the run starts
- stores, materializes, retains, and evicts prefixes only when full future
  knowledge says reuse will repay the cost under the benchmark's cost model

Why it is separate:
- this is not a reproduction of a named external runtime or paper system
- it is an evaluation oracle used to bound what an online policy could achieve

For this repo to count as correct:
- oracle must see the entire request sequence before the run
- oracle decisions must be evaluated with the same backend cost model used by
  the benchmarked system
- if cache capacity is limited, the oracle must solve the admission/eviction
  problem with future knowledge rather than a greedy local heuristic
- if offload, promotion, or transfer costs exist, the oracle must optimize
  under those same costs rather than assume free reuse

Minimum implementation bar:
- explicit future-trace pass
- explicit objective such as minimizing total latency or maximizing net reuse
  benefit
- capacity-aware materialization policy
- future-aware eviction / retention policy

What not to call this:
- a reactive cache with a small future-count heuristic
- a local lookahead approximation without full-trace access
- an external-system baseline

Separate runner:
- `run_oracle_future_reuse.py`
