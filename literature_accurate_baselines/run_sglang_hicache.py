from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
SRC = ROOT / "src"
for candidate in (ROOT, THIS_DIR, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from literature_accurate_baselines.adapter_lib import (
        ManagedServer,
        OpenAICompatClient,
        add_workload_args,
        build_requests,
        build_sglang_hicache_command,
        make_output_filename,
        maybe_sleep,
        normalize_api_base,
        parse_command_string,
        resolve_model,
        resolve_prompt_mode,
        save_summary,
        summarize_external_results,
    )
except ModuleNotFoundError:
    from adapter_lib import (
        ManagedServer,
        OpenAICompatClient,
        add_workload_args,
        build_requests,
        build_sglang_hicache_command,
        make_output_filename,
        maybe_sleep,
        normalize_api_base,
        parse_command_string,
        resolve_model,
        resolve_prompt_mode,
        save_summary,
        summarize_external_results,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_workload_args(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--request_endpoint", choices=["chat", "completion"], default="chat")
    parser.add_argument("--launch_server", action="store_true")
    parser.add_argument("--server_command", default=None, help="Optional full launch command string. Overrides the built SGLang HiCache command.")
    parser.add_argument("--python_executable", default="python")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--page_size", type=int, default=64)
    parser.add_argument("--hicache_ratio", type=float, default=2.0)
    parser.add_argument("--hicache_size", type=float, default=0.0)
    parser.add_argument("--hicache_io_backend", default="kernel")
    parser.add_argument("--hicache_write_policy", default="write_through")
    parser.add_argument("--hicache_mem_layout", default="page_first")
    parser.add_argument("--hicache_storage_backend", default=None)
    parser.add_argument("--hicache_storage_prefetch_policy", default=None)
    parser.add_argument("--hicache_storage_backend_extra_config", default=None)
    parser.add_argument("--server_extra_arg", action="append", default=[], help="Repeatable extra argument passed through to sglang.launch_server.")
    parser.add_argument("--server_ready_timeout_s", type=float, default=240.0)
    args = parser.parse_args()
    args.resolved_prompt_mode = resolve_prompt_mode(args)

    api_base = normalize_api_base(args.api_base, args.host, args.port)
    resolved_model = resolve_model(args.model) or args.model
    client = OpenAICompatClient(api_base=api_base, model=resolved_model, endpoint=args.request_endpoint)
    requests = build_requests(args)

    server = None
    if args.launch_server:
        command = parse_command_string(args.server_command) or build_sglang_hicache_command(args)
        out_dir = Path(args.output_dir)
        server = ManagedServer(
            command=command,
            cwd=str(Path.cwd()),
            env_updates={},
            stdout_path=out_dir / "sglang_hicache.stdout.txt",
            stderr_path=out_dir / "sglang_hicache.stderr.txt",
        )
        server.start()
    try:
        client.wait_until_ready(timeout_s=args.server_ready_timeout_s)
        results = []
        for idx, req in enumerate(requests):
            maybe_sleep(idx, requests, args.simulate_arrivals, args.max_arrival_sleep_ms)
            result = client.invoke(
                prompt=req.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.request_timeout_s,
            )
            result.request_id = req.request_id
            results.append(result)
    finally:
        if server is not None:
            server.stop()

    summary = {
        "SGLang_HiCache": summarize_external_results(results),
        "config": vars(args),
        "runtime": {
            "api_base": api_base,
            "resolved_model": resolved_model,
            "launch_server": bool(args.launch_server),
            "request_endpoint": args.request_endpoint,
        },
    }
    out_file = save_summary(args.output_dir, make_output_filename("benchmark_sglang_hicache", args), summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
