## Literature-Accurate Baselines

This folder is intentionally separate from the runnable benchmark harness.

Why:
- the main benchmark code should only contain baselines that are actually
  implemented inside this repository
- names like `KVFlow`, `SGLang HiCache`, and `LMCache` refer to broader runtime
  systems, not small heuristic drop-ins
- keeping them here avoids overstating what the main code reproduces

This folder is the place to stage:
- exact papers / system references
- integration notes
- reproduction requirements
- environment constraints
- future wrappers or adapters if you decide to benchmark them for real

Current status:
- no literature-accurate system integration is wired into `experiments/run_benchmark.py`
- the main code only benchmarks baselines implemented inside this repo
- runnable adapters now live in this folder, still separate from the main
  benchmark harness
- `oracle_future_reuse` is kept here as a custom offline evaluation oracle,
  not as a claim that this repo reproduces a named external system

Files:
- `oracle_future_reuse.md`
- `kvflow.md`
- `sglang_hicache.md`
- `lmcache.md`
- `run_oracle_future_reuse.py`
- `run_kvflow.py`
- `run_sglang_hicache.py`
- `run_lmcache.py`
- `adapter_lib.py`
- `oracle_engine.py`

Running:
- `run_oracle_future_reuse.py` executes the offline oracle against the repo's
  existing fake or Hugging Face backends.
- `run_sglang_hicache.py` can either attach to an existing SGLang HiCache
  server or launch one with official HiCache flags.
- `run_lmcache.py` can either attach to an LMCache-enabled vLLM/SGLang server
  or launch one with documented LMCache integration flags.
- `run_kvflow.py` replays a workflow trace against an external KVFlow-compatible
  serving endpoint. Because KVFlow is a workflow system rather than a generic
  off-the-shelf runtime flag, this adapter expects a real external deployment or
  a user-supplied launch command.

Primary sources:
- KVFlow paper: https://openreview.net/forum?id=5Iw1nDtYmT
- SGLang HiCache docs: https://docs.sglang.io/docs/advanced_features/hicache
- SGLang HiCache design: https://docs.sglang.io/docs/advanced_features/hicache_design
- LMCache docs: https://docs.lmcache.ai/
