## SGLang_HiCache

Target meaning:
- SGLang's hierarchical KV caching system layered on top of RadixAttention
- L1 GPU cache, L2 host-memory cache, and L3 distributed storage
- HiRadixTree metadata over cached KV spans
- local prefix match, L3-aware prefetch, and write-back / write-through cache
  management

Primary sources:
- https://docs.sglang.io/docs/advanced_features/hicache
- https://docs.sglang.io/docs/advanced_features/hicache_design

For this repo to count as literature-accurate:
- benchmark should use the real SGLang runtime or a faithful subsystem port
- cache tiers, metadata layout, prefetch behavior, and write-back policies
  should match the runtime
- transfer costs and serving pipeline behavior should come from the real system
- L3 behavior should reflect shared storage semantics rather than a local-only
  two-tier cache

Minimum implementation bar:
- real L1/L2/L3 hierarchical cache path
- HiRadixTree-style metadata or a faithful subsystem port
- runtime-consistent local match, prefetch, and write-back behavior
- runtime-consistent promotion / demotion and eviction

What not to call literature-accurate:
- a ShadowKV variant with faster promotion
- any heuristic that only mimics a two-tier cache idea inside this harness
- a GPU+CPU cache with no L3 or no runtime-consistent prefetch / write-back

Separate adapter:
- `run_sglang_hicache.py` launches or attaches to a real SGLang HiCache server
  through its OpenAI-compatible API path.
