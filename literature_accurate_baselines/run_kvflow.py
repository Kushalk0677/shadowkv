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
        load_trace_requests,
        maybe_sleep,
        normalize_api_base,
        parse_command_string,
        save_summary,
        summarize_external_results,
    )
except ModuleNotFoundError:
    from adapter_lib import (
        ManagedServer,
        OpenAICompatClient,
        load_trace_requests,
        maybe_sleep,
        normalize_api_base,
        parse_command_string,
        save_summary,
        summarize_external_results,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workflow_trace", required=True, help="JSON or JSONL trace replay file representing the external KVFlow workload order.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--request_endpoint", choices=["chat", "completion"], default="chat")
    parser.add_argument("--launch_server", action="store_true")
    parser.add_argument("--server_command", default=None, help="Required if you want this script to launch an external KVFlow-compatible server process.")
    parser.add_argument("--output_dir", default="results_external")
    parser.add_argument("--max_arrival_sleep_ms", type=float, default=500.0)
    parser.add_argument("--max_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request_timeout_s", type=float, default=600.0)
    parser.add_argument("--server_ready_timeout_s", type=float, default=240.0)
    args = parser.parse_args()

    api_base = normalize_api_base(args.api_base, args.host, args.port)
    client = OpenAICompatClient(api_base=api_base, model=args.model, endpoint=args.request_endpoint)
    requests = load_trace_requests(args.workflow_trace)

    server = None
    if args.launch_server:
        command = parse_command_string(args.server_command)
        if command is None:
            raise ValueError("--launch_server requires --server_command for KVFlow adapters")
        out_dir = Path(args.output_dir)
        server = ManagedServer(
            command=command,
            cwd=str(Path.cwd()),
            env_updates={},
            stdout_path=out_dir / "kvflow.stdout.txt",
            stderr_path=out_dir / "kvflow.stderr.txt",
        )
        server.start()
    try:
        client.wait_until_ready(timeout_s=args.server_ready_timeout_s)
        results = []
        for idx, req in enumerate(requests):
            maybe_sleep(idx, requests, True, args.max_arrival_sleep_ms)
            row = req.metadata or {}
            messages = row.get("messages")
            result = client.invoke(
                prompt=None if messages else req.prompt,
                messages=messages,
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
        "KVFlow": summarize_external_results(results),
        "config": vars(args),
        "runtime": {
            "api_base": api_base,
            "launch_server": bool(args.launch_server),
            "request_endpoint": args.request_endpoint,
            "trace_requests": len(requests),
        },
    }
    out_file = save_summary(args.output_dir, f"benchmark_kvflow_{args.model.replace('/', '_').replace(':', '_')}.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
