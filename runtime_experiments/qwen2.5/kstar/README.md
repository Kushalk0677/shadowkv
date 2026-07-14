# k-star Prefix Profile

This folder contains the k-star prefix-length profile used to support the runtime discussion.

## Files

| Path | Contents |
|---|---|
| `raw/prefix_profile_20260701/` | Primary Blackwell prefix profile with CSV, JSON, raw JSONL, summary, and metadata files. |
| `raw/response_usage_probe_20260701/` | Response-usage probe variant. |
| `raw/run_logs_20260701/` | Logs for the k-star and vLLM runtime runs. |

## Summary

The primary profile infers `k* = 16` cached tokens for Qwen2.5-1.5B, Qwen2.5-7B, and Qwen2.5-32B. Response-level cached-token fields stayed at zero in this vLLM path, so the relevant cache-hit signal is the vLLM metrics delta recorded in the profile outputs.

