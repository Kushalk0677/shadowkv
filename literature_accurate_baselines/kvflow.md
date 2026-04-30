## KVFlow

Target meaning:
- workflow-aware KV cache management framework for LLM-based multi-agent
  workflows
- models workflow execution with an Agent Step Graph
- assigns each agent a steps-to-execution score estimating how soon it will run
- uses that score for fine-grained eviction at the KV-node level in a
  tree-structured prefix cache
- proactively prefetches required KV tensors from CPU to GPU for agents
  scheduled in the next workflow step

Primary source:
- https://openreview.net/forum?id=5Iw1nDtYmT

For this repo to count as literature-accurate:
- the benchmark must expose an explicit workflow graph or equivalent execution
  structure, not just generic request grouping
- cache management must use a steps-to-execution style signal derived from that
  workflow structure
- eviction must operate at KV-node granularity over a shared prefix tree rather
  than only at whole-prefix or whole-request granularity
- prefetch must reflect KVFlow's workflow-aware next-step prefetch behavior,
  including CPU-to-GPU movement for upcoming agents
- evaluation should document exactly what online workflow information is
  available to the baseline

Minimum implementation bar:
- explicit Agent Step Graph or faithful equivalent
- explicit steps-to-execution computation
- KV-node-level eviction policy tied to that signal
- workflow-aware prefetch path for near-future agents

What not to call literature-accurate:
- a generic future-aware heuristic without an execution-graph model
- a frequency-based heuristic with no workflow structure
- a simple prefix admission rule with no KV-node-level eviction
- prefetch that is not tied to the next workflow step

Separate adapter:
- `run_kvflow.py` replays a workflow trace against an external KVFlow-capable
  server endpoint. It is intentionally separate from the main benchmark harness.
