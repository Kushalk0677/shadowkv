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
        ExternalAdmissionController,
        ManagedServer,
        OpenAICompatClient,
        add_workload_args,
        build_lmcache_command,
        build_requests,
        build_sglang_radix_attention_command,
        build_vllm_apc_command,
        make_output_filename,
        maybe_sleep,
        normalize_api_base,
        parse_command_string,
        reset_runtime_cache,
        resolve_model,
        resolve_prompt_mode,
        save_summary,
        summarize_external_results,
    )
except ModuleNotFoundError:
    from adapter_lib import (
        ExternalAdmissionController,
        ManagedServer,
        OpenAICompatClient,
        add_workload_args,
        build_lmcache_command,
        build_requests,
        build_sglang_radix_attention_command,
        build_vllm_apc_command,
        make_output_filename,
        maybe_sleep,
        normalize_api_base,
        parse_command_string,
        reset_runtime_cache,
        resolve_model,
        resolve_prompt_mode,
        save_summary,
        summarize_external_results,
    )


BASELINE_CHOICES = (
    "vllm_apc",
    "vllm_apc_shadowkv_plus",
    "sglang_radix_attention",
    "sglang_radix_attention_shadowkv_plus",
    "lmcache",
    "lmcache_shadowkv_plus",
)


def _runtime_kind(args: argparse.Namespace) -> str:
    if args.baseline.startswith("sglang"):
        return "sglang"
    if args.baseline.startswith("lmcache") and args.lmcache_engine == "sglang":
        return "sglang"
    return "vllm"


def _uses_admission_controller(args: argparse.Namespace) -> bool:
    return args.baseline.endswith("_shadowkv_plus")


def _build_launch(args: argparse.Namespace):
    if args.baseline.startswith("vllm_apc"):
        return build_vllm_apc_command(args), {}
    if args.baseline.startswith("sglang_radix_attention"):
        return build_sglang_radix_attention_command(args), {}
    if args.baseline.startswith("lmcache"):
        args.engine = args.lmcache_engine
        return build_lmcache_command(args)
    raise ValueError(f"Unsupported baseline: {args.baseline}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run literature-accurate external runtime cache baselines.")
    parser.add_argument("--baseline", choices=BASELINE_CHOICES, required=True)
    add_workload_args(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--request_endpoint", choices=["auto", "chat", "completion"], default="auto")
    parser.add_argument("--launch_server", action="store_true")
    parser.add_argument("--server_command", default=None, help="Optional full launch command string. Overrides the built runtime command.")
    parser.add_argument("--python_executable", default="python")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention_backend", default=None)
    parser.add_argument("--lmcache_engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--lmcache_mode", choices=["transfer", "offload"], default="transfer")
    parser.add_argument("--lmcache_chunk_size", type=int, default=256)
    parser.add_argument("--lmcache_config_file", default=None)
    parser.add_argument("--kv_offloading_size", type=float, default=10.0)
    parser.add_argument("--server_extra_arg", action="append", default=[], help="Repeatable extra argument passed through to the runtime.")
    parser.add_argument("--server_ready_timeout_s", type=float, default=240.0)
    parser.add_argument("--reset_external_cache", action="store_true", help="When resetting vLLM prefix cache, also reset connector-managed external cache.")
    args = parser.parse_args()
    args.resolved_prompt_mode = resolve_prompt_mode(args)

    runtime = _runtime_kind(args)
    if args.port is None:
        args.port = 30000 if runtime == "sglang" else 8000
    endpoint = args.request_endpoint
    if endpoint == "auto":
        endpoint = "completion" if runtime == "vllm" else "chat"

    api_base = normalize_api_base(args.api_base, args.host, args.port)
    resolved_model = resolve_model(args.model) or args.model
    client = OpenAICompatClient(api_base=api_base, model=resolved_model, endpoint=endpoint)
    requests = build_requests(args)

    server = None
    if args.launch_server:
        command = parse_command_string(args.server_command)
        env_updates = {}
        if command is None:
            command, env_updates = _build_launch(args)
        out_dir = Path(args.output_dir)
        server = ManagedServer(
            command=command,
            cwd=str(Path.cwd()),
            env_updates=env_updates,
            stdout_path=out_dir / f"{args.baseline}.stdout.txt",
            stderr_path=out_dir / f"{args.baseline}.stderr.txt",
        )
        server.start()

    admission = ExternalAdmissionController(resolved_model, runtime=runtime) if _uses_admission_controller(args) else None
    admission_metrics = {
        "admission_controller_enabled": bool(admission is not None),
        "admission_plans_total": 0,
        "admission_allow_total": 0,
        "admission_bypass_total": 0,
        "admission_runtime_cache_resets": 0,
        "admission_policy_net_utility_ms": 0.0,
    }

    try:
        client.wait_until_ready(timeout_s=args.server_ready_timeout_s)
        results = []
        shared_prefix_token_cache = {}
        for idx, req in enumerate(requests):
            maybe_sleep(idx, requests, args.simulate_arrivals, args.max_arrival_sleep_ms)
            metadata = dict(req.metadata or {})
            metadata["arrival_time"] = req.arrival_time
            plan = None
            tokens = ()
            if admission is not None:
                shared_prefix_text = metadata.get("shared_prefix_text")
                if shared_prefix_text:
                    hint_len = shared_prefix_token_cache.get(shared_prefix_text)
                    if hint_len is None:
                        hint_len = len(admission.tokenize(str(shared_prefix_text)))
                        shared_prefix_token_cache[shared_prefix_text] = hint_len
                    metadata["shared_prefix_hint_tokens"] = hint_len
                plan, tokens = admission.plan(req.prompt, metadata=metadata)
                admission_metrics["admission_plans_total"] += 1
                admission_metrics["admission_policy_net_utility_ms"] += float(plan.score)
                if plan.strategy == "bypass":
                    admission_metrics["admission_bypass_total"] += 1
                    if reset_runtime_cache(client, runtime, reset_external=args.reset_external_cache or args.baseline.startswith("lmcache")):
                        admission_metrics["admission_runtime_cache_resets"] += 1
                else:
                    admission_metrics["admission_allow_total"] += 1
            result = client.invoke(
                prompt=req.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.request_timeout_s,
            )
            result.request_id = req.request_id
            results.append(result)
            if admission is not None and plan is not None:
                stored = admission.record_after_request(tokens, plan, metadata=metadata)
                if plan.strategy == "bypass" and reset_runtime_cache(client, runtime, reset_external=args.reset_external_cache or args.baseline.startswith("lmcache")):
                    admission_metrics["admission_runtime_cache_resets"] += 1
                if stored:
                    admission_metrics["store_successes"] = int(admission_metrics.get("store_successes", 0)) + 1
    finally:
        if server is not None:
            server.stop()

    summary_metrics = summarize_external_results(results)
    summary_metrics.update(admission_metrics)
    summary = {
        args.baseline: summary_metrics,
        "config": vars(args),
        "runtime": {
            "runtime_kind": runtime,
            "api_base": api_base,
            "resolved_model": resolved_model,
            "launch_server": bool(args.launch_server),
            "request_endpoint": endpoint,
            "literature_accurate_external_runtime": True,
        },
    }
    out_file = save_summary(args.output_dir, make_output_filename(f"benchmark_{args.baseline}", args), summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
