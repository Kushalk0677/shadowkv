# Operator notes - 2026-07-10

The received ZIP used Windows backslashes in every archive member name. It was
extracted through a path-normalizing ZIP reader before transfer to first-light.

One model identifier was corrected without changing the experiment matrix or
policy:

- received: `google/gemma-4-12b-it`
- official: `google/gemma-4-12B-it`

Three legacy dataset IDs were updated to the namespaced repositories already
validated by the earlier Blackwell harness:

- `ag_news` -> `fancyzhx/ag_news`
- `cnn_dailymail` -> `abisee/cnn_dailymail`
- `xsum` -> `EdinburghNLP/xsum`

The official model configuration resolves as `gemma4_unified` under
Transformers 5.10.2. The target run is limited to:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-4-12B-it`

Each model will be smoke-tested before the full 10-dataset, 128-request,
three-engine semantic sweep.

The first Qwen 7B smoke exposed a long-scaffold admission mismatch: a 460-token
hint was capped to a 128-token cache entry, but the reuse gate compared the hit
against the uncapped hint and bypassed it. Scaffold admission now compares
against the effective cacheable hint. A deterministic oversized-scaffold
regression test covers this case.

The first Gemma 4 smoke then exposed mutable-cache corruption: two successful
reuses were followed by a backend fallback because Gemma hybrid attention had
updated tensors shared with the canonical cache entry. HF cache preparation now
deep-copies the cached state before each measured reuse. A regression test
verifies that mutations to the prepared cache cannot alter canonical tensors.
