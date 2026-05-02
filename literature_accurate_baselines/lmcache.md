## LMCache

Target meaning:
- a reusable KV cache layer for prefill-once, reuse-everywhere serving
- supports reuse of cached text not necessarily limited to shared prefixes
- supports KV offloading, sharing across engine instances, and multiple storage
  backends with measured transfer / reload costs

Primary source:
- https://docs.lmcache.ai/

For this repo to count as literature-accurate:
- benchmark should exercise the real LMCache system or a faithful subsystem
  port
- CPU/GPU transfer and reuse overheads should come from the implemented system
- admission, lookup, sharing, and eviction behavior should match the actual
  cache-store design
- if the benchmark claims LMCache behavior, it should account for non-prefix
  reuse and cross-instance sharing rather than only prefix reuse inside one
  worker

Minimum implementation bar:
- explicit KV store layer
- explicit offload / reload path
- measured transfer overheads
- documented admission / eviction semantics
- explicit reuse lookup semantics across the supported cache scope

What not to call literature-accurate:
- a reactive prefix cache with a constant extra latency penalty
- any baseline that only approximates offload with synthetic overheads
- a prefix-only cache that ignores LMCache's broader reusable-text semantics

Separate adapter:
- `run_lmcache.py` launches or attaches to a real LMCache-backed vLLM or
  SGLang server through its OpenAI-compatible API path.
