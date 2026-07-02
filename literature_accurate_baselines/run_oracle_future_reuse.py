from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from literature_accurate_baselines.adapter_lib import (
        add_workload_args,
        build_requests,
        make_output_filename,
        maybe_sleep,
        resolve_model,
        resolve_prompt_mode,
        save_summary,
    )
    from literature_accurate_baselines.oracle_engine import OracleFutureReuseEngine
except ModuleNotFoundError:
    from adapter_lib import (
        add_workload_args,
        build_requests,
        make_output_filename,
        maybe_sleep,
        resolve_model,
        resolve_prompt_mode,
        save_summary,
    )
    from oracle_engine import OracleFutureReuseEngine
from proactive_kv_cache.engines import maybe_shutdown, summarize_engine
from proactive_kv_cache.models import load_backend
from proactive_kv_cache.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["fake", "hf"], default="hf")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_memory_mb", type=int, default=64)
    add_workload_args(parser)
    args = parser.parse_args()
    args.resolved_prompt_mode = resolve_prompt_mode(args)

    set_seed(args.seed)
    requests = build_requests(args)
    backend = load_backend(
        args.backend,
        model_name=resolve_model(args.model),
        device=args.device,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )

    tokenized_trace = []
    shared_prefix_token_cache = {}
    for req in requests:
        tokens = backend.tokenize(req.prompt)
        metadata = dict(req.metadata or {})
        metadata["arrival_time"] = req.arrival_time
        shared_prefix_text = metadata.get("shared_prefix_text")
        if shared_prefix_text:
            hint_len = shared_prefix_token_cache.get(shared_prefix_text)
            if hint_len is None:
                hint_len = len(backend.tokenize(shared_prefix_text))
                shared_prefix_token_cache[shared_prefix_text] = hint_len
            metadata["shared_prefix_hint_tokens"] = min(int(hint_len), len(tokens))
        tokenized_trace.append((tuple(tokens), metadata))

    engine = OracleFutureReuseEngine(backend=backend, request_trace=tokenized_trace, max_memory_mb=args.max_memory_mb)
    for idx, req in enumerate(requests):
        maybe_sleep(idx, requests, args.simulate_arrivals, args.max_arrival_sleep_ms)
        tokens, metadata = tokenized_trace[idx]
        engine.serve_tokens(req.request_id, tokens, metadata=metadata)
    maybe_shutdown(engine)

    summary = {
        "oracle_future_reuse": summarize_engine(engine),
        "config": vars(args),
        "capabilities": {
            "supports_external_kv": getattr(backend, "supports_external_kv", False),
            "supports_native_prefix_caching": getattr(backend, "supports_native_prefix_caching", False),
        },
    }
    summary["config"]["resolved_model"] = resolve_model(args.model)
    out_file = save_summary(args.output_dir, make_output_filename("benchmark_oracle_future_reuse", args), summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
