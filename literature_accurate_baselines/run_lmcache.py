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
        build_lmcache_command,
        build_requests,
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
        build_lmcache_command,
        build_requests,
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
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--request_endpoint", choices=["auto", "chat", "completion"], default="auto")
    parser.add_argument("--launch_server", action="store_true")
    parser.add_argument("--server_command", default=None, help="Optional full launch command string. Overrides the built LMCache command.")
    parser.add_argument("--python_executable", default="python")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--lmcache_mode", choices=["transfer", "offload"], default="transfer")
    parser.add_argument("--lmcache_chunk_size", type=int, default=256)
    parser.add_argument("--lmcache_config_file", default=None)
    parser.add_argument("--kv_offloading_size", type=float, default=10.0)
    parser.add_argument("--server_extra_arg", action="append", default=[], help="Repeatable extra argument passed through to the runtime.")
    parser.add_argument("--server_ready_timeout_s", type=float, default=240.0)
    args = parser.parse_args()
    args.resolved_prompt_mode = resolve_prompt_mode(args)

    api_base = normalize_api_base(args.api_base, args.host, args.port)
    resolved_model = resolve_model(args.model) or args.model
    endpoint = args.request_endpoint
    if endpoint == "auto":
        endpoint = "completion" if args.engine == "vllm" else "chat"
    client = OpenAICompatClient(api_base=api_base, model=resolved_model, endpoint=endpoint)
    requests = build_requests(args)

    server = None
    if args.launch_server:
        command = parse_command_string(args.server_command)
        env_updates = {}
        if command is None:
            command, env_updates = build_lmcache_command(args)
        out_dir = Path(args.output_dir)
        server = ManagedServer(
            command=command,
            cwd=str(Path.cwd()),
            env_updates=env_updates,
            stdout_path=out_dir / "lmcache.stdout.txt",
            stderr_path=out_dir / "lmcache.stderr.txt",
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
        "LMCache": summarize_external_results(results),
        "config": vars(args),
        "runtime": {
            "engine": args.engine,
            "api_base": api_base,
            "resolved_model": resolved_model,
            "launch_server": bool(args.launch_server),
            "request_endpoint": endpoint,
        },
    }
    out_file = save_summary(args.output_dir, make_output_filename("benchmark_lmcache", args), summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
