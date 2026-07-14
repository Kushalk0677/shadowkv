from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
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
        vllm_compat_env_updates,
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
        vllm_compat_env_updates,
    )


BASELINE_CHOICES = (
    "vllm_apc",
    "vllm_apc_shadowkv_plus",
    "sglang_radix_attention",
    "sglang_radix_attention_shadowkv_plus",
    "lmcache",
    "lmcache_shadowkv_plus",
)

ADMISSION_PRESETS = (
    "conservative",
    "balanced",
    "aggressive_prefix",
    "low_latency",
)

DEFAULT_ADMISSION_TUNING_PRESETS = (
    "balanced",
    "aggressive_prefix",
    "low_latency",
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
        return build_vllm_apc_command(args), vllm_compat_env_updates()
    if args.baseline.startswith("sglang_radix_attention"):
        return build_sglang_radix_attention_command(args), {}
    if args.baseline.startswith("lmcache"):
        args.engine = args.lmcache_engine
        return build_lmcache_command(args)
    raise ValueError(f"Unsupported baseline: {args.baseline}")


def _apply_admission_preset(controller: ExternalAdmissionController, preset: str) -> ExternalAdmissionController:
    preset = str(preset or "balanced")
    if preset not in ADMISSION_PRESETS:
        raise ValueError(f"Unknown admission preset: {preset}")
    controller.admission_preset = preset
    if preset == "conservative":
        controller.full_ms_per_token = 0.30
        controller.reuse_overhead_ms = 2.0
        controller.bank.min_match_length = max(int(controller.bank.min_match_length), 16)
    elif preset == "aggressive_prefix":
        controller.full_ms_per_token = 0.45
        controller.reuse_overhead_ms = 0.55
        controller.bank.min_match_length = min(int(controller.bank.min_match_length), 8)
    elif preset == "low_latency":
        controller.full_ms_per_token = 0.50
        controller.reuse_overhead_ms = 0.25
        controller.bank.min_match_length = min(int(controller.bank.min_match_length), 6)
    return controller


def _new_admission_controller(model: str, runtime: str, preset: str) -> ExternalAdmissionController:
    controller = ExternalAdmissionController(model, runtime=runtime)
    return _apply_admission_preset(controller, preset)


def _runtime_admission_score(metrics: dict, metric: str) -> float:
    if metric == "mean_latency_ms":
        return float(metrics.get("mean_latency_ms", 0.0))
    if metric == "p95_latency_ms":
        return float(metrics.get("p95_latency_ms", 0.0))
    if metric == "cached_adjusted_latency":
        mean_latency = float(metrics.get("mean_latency_ms", 0.0))
        p95_latency = float(metrics.get("p95_latency_ms", mean_latency))
        cached_mean = float(metrics.get("cached_tokens_mean", 0.0))
        return mean_latency + 0.05 * p95_latency - 0.02 * cached_mean
    raise ValueError(f"Unsupported admission tuning metric: {metric}")


def _prepare_admission_metadata(admission: ExternalAdmissionController, req, shared_prefix_token_cache: dict[str, int]) -> dict:
    metadata = dict(req.metadata or {})
    metadata["arrival_time"] = req.arrival_time
    shared_prefix_text = metadata.get("shared_prefix_text")
    if shared_prefix_text:
        hint_len = shared_prefix_token_cache.get(shared_prefix_text)
        if hint_len is None:
            hint_len = len(admission.tokenize(str(shared_prefix_text)))
            shared_prefix_token_cache[shared_prefix_text] = hint_len
        metadata["shared_prefix_hint_tokens"] = hint_len
    return metadata


def _wait_until_ready_or_server_exit(client: OpenAICompatClient, server: ManagedServer | None, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        if server is not None:
            rc = server.poll()
            if rc is not None:
                logs = server.tail_logs(max_lines=240)
                detail = f"\n{logs}" if logs else ""
                raise RuntimeError(f"Runtime server exited before becoming ready (rc={rc}).{detail}")
        for path in ("/v1/models", "/health"):
            try:
                req = urllib.request.Request(client.api_base + path, method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if 200 <= resp.status < 300:
                        return
            except Exception as exc:
                last_error = str(exc)
        time.sleep(1.0)
    logs = server.tail_logs(max_lines=240) if server is not None else ""
    detail = f"\n{logs}" if logs else ""
    raise RuntimeError(
        f"Server at {client.api_base} did not become ready within {timeout_s:.0f}s. "
        f"Last error: {last_error}{detail}"
    )


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
    parser.add_argument(
        "--admission_mode",
        choices=["strict_no_write", "write_through_admission"],
        default="write_through_admission",
        help="How MeritKV admission enforces bypasses for external runtime caches.",
    )
    parser.add_argument("--admission_preset", choices=ADMISSION_PRESETS, default="balanced", help="Fixed external admission preset used for the measured run.")
    parser.add_argument("--enable_admission_tuning", action="store_true", help="Run a short unmeasured preset calibration before the measured runtime run.")
    parser.add_argument("--admission_tuning_requests", type=int, default=16, help="Number of leading requests used for unmeasured runtime admission calibration.")
    parser.add_argument("--admission_tuning_presets", nargs="+", choices=ADMISSION_PRESETS, default=list(DEFAULT_ADMISSION_TUNING_PRESETS), help="Candidate admission presets for --enable_admission_tuning.")
    parser.add_argument("--admission_tuning_metric", choices=["mean_latency_ms", "p95_latency_ms", "cached_adjusted_latency"], default="cached_adjusted_latency", help="Metric minimized during admission preset calibration.")
    args = parser.parse_args()
    args.resolved_prompt_mode = resolve_prompt_mode(args)
    args.admission_tuning_report = None

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

    admission = _new_admission_controller(resolved_model, runtime, args.admission_preset) if _uses_admission_controller(args) else None
    admission_metrics = {
        "admission_controller_enabled": bool(admission is not None),
        "admission_plans_total": 0,
        "admission_allow_total": 0,
        "admission_bypass_total": 0,
        "admission_runtime_cache_resets": 0,
        "admission_pre_request_cache_resets": 0,
        "admission_post_request_cache_resets": 0,
        "admission_runtime_cache_reset_failures": 0,
        "admission_policy_net_utility_ms": 0.0,
        "admission_write_through_bypass_total": 0,
        "admission_bypass_store_successes": 0,
        "admission_selected_preset": args.admission_preset,
        "admission_calibrated_full_ms_per_token": 0.0,
        "admission_calibrated_reuse_overhead_ms": 0.0,
    }

    try:
        _wait_until_ready_or_server_exit(client, server, args.server_ready_timeout_s)
        reset_external = args.reset_external_cache or args.baseline.startswith("lmcache")
        if admission is not None and args.enable_admission_tuning:
            tuning_requests = requests[: max(int(args.admission_tuning_requests), 0)]
            candidates = []
            for preset in args.admission_tuning_presets:
                if preset not in candidates:
                    candidates.append(preset)
            tuning_rows = []
            selected_preset = args.admission_preset
            best_score = None
            for preset in candidates:
                reset_runtime_cache(client, runtime, reset_external=reset_external)
                candidate_admission = _new_admission_controller(resolved_model, runtime, preset)
                candidate_results = []
                shared_prefix_token_cache = {}
                for req in tuning_requests:
                    metadata = _prepare_admission_metadata(candidate_admission, req, shared_prefix_token_cache)
                    plan, tokens = candidate_admission.plan(req.prompt, metadata=metadata)
                    result = client.invoke(
                        prompt=req.prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout_s=args.request_timeout_s,
                    )
                    result.request_id = req.request_id
                    candidate_results.append(result)
                    candidate_admission.record_after_request(
                        tokens,
                        plan,
                        metadata=metadata,
                        result=result,
                        allow_bypass_store=args.admission_mode == "write_through_admission",
                    )
                metrics = summarize_external_results(candidate_results)
                metrics["admission_calibrated_full_ms_per_token"] = float(candidate_admission.full_ms_per_token)
                metrics["admission_calibrated_reuse_overhead_ms"] = float(candidate_admission.reuse_overhead_ms)
                score = _runtime_admission_score(metrics, args.admission_tuning_metric)
                row = {"preset": preset, "score": score, "metrics": metrics}
                tuning_rows.append(row)
                if best_score is None or score < best_score:
                    best_score = score
                    selected_preset = preset
            reset_runtime_cache(client, runtime, reset_external=reset_external)
            args.admission_preset = selected_preset
            admission_metrics["admission_selected_preset"] = selected_preset
            admission = _new_admission_controller(resolved_model, runtime, selected_preset)
            report = {
                "enabled": True,
                "method": "fixed_preset_calibration",
                "selected_preset": selected_preset,
                "selection_metric": args.admission_tuning_metric,
                "tuning_requests": len(tuning_requests),
                "candidate_presets": candidates,
                "candidates": tuning_rows,
                "excluded_from_measured_run": True,
            }
            report_file = Path(args.output_dir) / "admission_tuning_report.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["report_file"] = str(report_file)
            args.admission_tuning_report = report

        results = []
        shared_prefix_token_cache = {}
        for idx, req in enumerate(requests):
            maybe_sleep(idx, requests, args.simulate_arrivals, args.max_arrival_sleep_ms)
            plan = None
            tokens = ()
            bypass_runtime_cache = False
            if admission is not None:
                metadata = _prepare_admission_metadata(admission, req, shared_prefix_token_cache)
                plan, tokens = admission.plan(req.prompt, metadata=metadata)
                admission_metrics["admission_plans_total"] += 1
                admission_metrics["admission_policy_net_utility_ms"] += float(plan.score)
                if plan.strategy == "bypass":
                    admission_metrics["admission_bypass_total"] += 1
                    if args.admission_mode == "strict_no_write":
                        bypass_runtime_cache = True
                        if reset_runtime_cache(client, runtime, reset_external=reset_external):
                            admission_metrics["admission_runtime_cache_resets"] += 1
                            admission_metrics["admission_pre_request_cache_resets"] += 1
                        else:
                            admission_metrics["admission_runtime_cache_reset_failures"] += 1
                    else:
                        admission_metrics["admission_write_through_bypass_total"] += 1
                else:
                    admission_metrics["admission_allow_total"] += 1
            else:
                metadata = dict(req.metadata or {})
                metadata["arrival_time"] = req.arrival_time
            result = client.invoke(
                prompt=req.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.request_timeout_s,
            )
            result.request_id = req.request_id
            results.append(result)
            if bypass_runtime_cache:
                if reset_runtime_cache(client, runtime, reset_external=reset_external):
                    admission_metrics["admission_runtime_cache_resets"] += 1
                    admission_metrics["admission_post_request_cache_resets"] += 1
                else:
                    admission_metrics["admission_runtime_cache_reset_failures"] += 1
            if admission is not None and plan is not None:
                stored = admission.record_after_request(
                    tokens,
                    plan,
                    metadata=metadata,
                    result=result,
                    allow_bypass_store=args.admission_mode == "write_through_admission",
                )
                if stored:
                    admission_metrics["store_successes"] = int(admission_metrics.get("store_successes", 0)) + 1
                    if plan.strategy == "bypass":
                        admission_metrics["admission_bypass_store_successes"] += 1
                admission_metrics["admission_calibrated_full_ms_per_token"] = float(admission.full_ms_per_token)
                admission_metrics["admission_calibrated_reuse_overhead_ms"] = float(admission.reuse_overhead_ms)
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
