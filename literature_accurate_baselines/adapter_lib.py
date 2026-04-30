from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proactive_kv_cache.datasets import list_datasets, list_prompt_modes
from proactive_kv_cache.metrics import summarize_run
from proactive_kv_cache.utils import set_seed
from proactive_kv_cache.workload import SYNTHETIC_VARIANTS, Request, make_public_dataset_workload, make_synthetic_workload


MODEL_PRESETS = {
    "tiny": "sshleifer/tiny-gpt2",
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
    "phi3mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama32_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen25_15b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def resolve_model(model: str | None) -> str | None:
    if model is None:
        return None
    return MODEL_PRESETS.get(model, model)


def resolve_prompt_mode(args: argparse.Namespace) -> str:
    if args.prompt_mode != "auto":
        return args.prompt_mode
    if args.workload == "public_dataset":
        return "templated"
    return "raw"


def maybe_sleep(current_idx: int, requests: Sequence[Request], simulate_arrivals: bool, max_sleep_ms: float) -> None:
    if not simulate_arrivals or current_idx == 0:
        return
    delay = requests[current_idx].arrival_time - requests[current_idx - 1].arrival_time
    if delay > 0:
        time.sleep(min(delay, max_sleep_ms / 1000.0))


def build_requests(args: argparse.Namespace) -> List[Request]:
    prompt_mode = getattr(args, "resolved_prompt_mode", "raw")
    if args.workload == "synthetic":
        return make_synthetic_workload(
            variant=args.variant,
            n_requests=args.n_requests,
            seed=args.seed,
            mean_inter_arrival_ms=args.mean_inter_arrival_ms,
        )
    if args.workload == "public_dataset":
        return make_public_dataset_workload(
            dataset_name=args.dataset,
            split=args.dataset_split,
            n_requests=args.n_requests,
            seed=args.seed,
            mean_inter_arrival_ms=args.mean_inter_arrival_ms or 150.0,
            prompt_mode=prompt_mode,
        )
    raise ValueError(f"Unsupported workload: {args.workload}")


def add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", choices=["synthetic", "public_dataset"], default="public_dataset")
    parser.add_argument("--variant", choices=sorted(SYNTHETIC_VARIANTS.keys()), default="high_skew")
    parser.add_argument("--dataset", choices=list_datasets(), default="daily_dialog")
    parser.add_argument("--prompt_mode", choices=["auto", *list_prompt_modes()], default="auto")
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--n_requests", type=int, default=64)
    parser.add_argument("--simulate_arrivals", dest="simulate_arrivals", action="store_true")
    parser.add_argument("--disable_arrival_simulation", dest="simulate_arrivals", action="store_false")
    parser.set_defaults(simulate_arrivals=True)
    parser.add_argument("--mean_inter_arrival_ms", type=float, default=50.0)
    parser.add_argument("--max_arrival_sleep_ms", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results_external")
    parser.add_argument("--max_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request_timeout_s", type=float, default=600.0)


def normalize_api_base(api_base: str | None, host: str, port: int) -> str:
    if api_base:
        return api_base.rstrip("/")
    return f"http://{host}:{port}"


def model_slug(model: str | None) -> str:
    return (resolve_model(model) or model or "default").replace("/", "_").replace(":", "_").replace(".", "_")


def workload_slug(args: argparse.Namespace) -> str:
    if args.workload == "public_dataset":
        return f"{args.dataset}_{args.resolved_prompt_mode}"
    return args.variant


def save_summary(output_dir: str, filename: str, payload: Dict[str, Any]) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def make_output_filename(prefix: str, args: argparse.Namespace) -> str:
    return f"{prefix}_{model_slug(args.model)}_{args.workload}_{workload_slug(args)}.json"


@dataclass
class ExternalCallResult:
    request_id: int
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


def summarize_external_results(results: Sequence[ExternalCallResult]) -> Dict[str, Any]:
    latencies = [result.latency_ms for result in results]
    summary = summarize_run(
        latencies=latencies,
        gpu_utils=[None] * len(results),
        bank_metrics={},
        speculative_precomputes=0,
        speculative_cost_ms=0.0,
        total_wall_time_s=max(sum(latencies) / 1000.0, 1e-9),
        extra_metrics={
            "requests_seen": len(results),
            "reuse_attempts": 0,
            "reuse_successes": 0,
            "reuse_failures": 0,
            "reuse_backend_fallbacks": 0,
            "store_attempts": 0,
            "store_successes": 0,
            "store_skips": 0,
            "bypassed_matches": 0,
            "reused_prefix_tokens_total": 0,
            "recompute_tokens_total": 0,
            "estimated_tokens_saved_total": 0,
            "saved_latency_estimate_ms": 0.0,
            "store_latency_total_ms": 0.0,
            "full_prefill_latency_total_ms": 0.0,
        },
    ).to_dict()
    summary["prompt_tokens_total"] = int(sum(result.prompt_tokens for result in results))
    summary["completion_tokens_total"] = int(sum(result.completion_tokens for result in results))
    summary["total_tokens_total"] = int(sum(result.total_tokens for result in results))
    summary["cached_tokens_total"] = int(sum(result.cached_tokens for result in results))
    summary["cached_tokens_mean"] = float(summary["cached_tokens_total"] / max(len(results), 1))
    return summary


class OpenAICompatClient:
    def __init__(self, api_base: str, model: str, endpoint: str = "chat") -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.endpoint = endpoint

    def wait_until_ready(self, timeout_s: float = 180.0, poll_s: float = 1.0) -> None:
        deadline = time.time() + timeout_s
        last_error: str | None = None
        while time.time() < deadline:
            for path in ("/v1/models", "/health"):
                try:
                    req = urllib.request.Request(self.api_base + path, method="GET")
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        if 200 <= resp.status < 300:
                            return
                except Exception as exc:  # pragma: no cover - runtime path
                    last_error = str(exc)
            time.sleep(poll_s)
        raise RuntimeError(f"Server at {self.api_base} did not become ready within {timeout_s:.0f}s. Last error: {last_error}")

    def _post_json(self, path: str, body: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.api_base + path,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw)

    def invoke(
        self,
        *,
        prompt: str | None = None,
        messages: Sequence[Dict[str, str]] | None = None,
        max_tokens: int = 1,
        temperature: float = 0.0,
        timeout_s: float = 600.0,
        extra_body: Dict[str, Any] | None = None,
    ) -> ExternalCallResult:
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            body.update(extra_body)
        if self.endpoint == "chat":
            body["messages"] = list(messages) if messages is not None else [{"role": "user", "content": prompt or ""}]
            path = "/v1/chat/completions"
        elif self.endpoint == "completion":
            if prompt is None and messages is not None:
                prompt = "\n".join(str(item.get("content", "")) for item in messages)
            body["prompt"] = prompt or ""
            path = "/v1/completions"
        else:
            raise ValueError(f"Unsupported endpoint: {self.endpoint}")

        start = time.perf_counter()
        payload = self._post_json(path, body, timeout_s=timeout_s)
        latency_ms = (time.perf_counter() - start) * 1000.0

        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = int(prompt_details.get("cached_tokens") or 0)
        return ExternalCallResult(
            request_id=-1,
            latency_ms=float(latency_ms),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )


class ManagedServer:
    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | None,
        env_updates: Dict[str, str] | None,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        self.command = list(command)
        self.cwd = cwd
        self.env_updates = dict(env_updates or {})
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.process: subprocess.Popen[str] | None = None
        self._stdout_handle = None
        self._stderr_handle = None

    def start(self) -> None:
        env = os.environ.copy()
        env.update(self.env_updates)
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout_handle = self.stdout_path.open("w", encoding="utf-8")
        self._stderr_handle = self.stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=env,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
        )

    def poll(self) -> int | None:
        if self.process is None:
            return None
        return self.process.poll()

    def stop(self) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=20.0)
                except subprocess.TimeoutExpired:  # pragma: no cover - runtime path
                    self.process.kill()
                    self.process.wait(timeout=5.0)
        finally:
            if self._stdout_handle is not None:
                self._stdout_handle.close()
            if self._stderr_handle is not None:
                self._stderr_handle.close()


def parse_command_string(command: str | None) -> List[str] | None:
    if not command:
        return None
    return shlex.split(command)


def build_sglang_hicache_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        args.python_executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        resolve_model(args.model) or args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--page-size",
        str(args.page_size),
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        str(args.hicache_ratio),
        "--hicache-size",
        str(args.hicache_size),
        "--hicache-io-backend",
        args.hicache_io_backend,
        "--hicache-write-policy",
        args.hicache_write_policy,
        "--enable-cache-report",
        "--enable-metrics",
    ]
    if args.tp and args.tp > 1:
        cmd.extend(["--tp", str(args.tp)])
    if args.hicache_mem_layout:
        cmd.extend(["--hicache-mem-layout", args.hicache_mem_layout])
    if args.hicache_storage_backend:
        cmd.extend(["--hicache-storage-backend", args.hicache_storage_backend])
    if args.hicache_storage_prefetch_policy:
        cmd.extend(["--hicache-storage-prefetch-policy", args.hicache_storage_prefetch_policy])
    if args.hicache_storage_backend_extra_config:
        cmd.extend(["--hicache-storage-backend-extra-config", args.hicache_storage_backend_extra_config])
    for extra in args.server_extra_arg:
        cmd.append(extra)
    return cmd


def build_lmcache_command(args: argparse.Namespace) -> tuple[List[str], Dict[str, str]]:
    env_updates: Dict[str, str] = {}
    if args.lmcache_chunk_size:
        env_updates["LMCACHE_CHUNK_SIZE"] = str(args.lmcache_chunk_size)
    if args.lmcache_config_file:
        env_updates["LMCACHE_CONFIG_FILE"] = args.lmcache_config_file

    model_name = resolve_model(args.model) or args.model
    if args.engine == "vllm":
        cmd = [
            "vllm",
            "serve",
            model_name,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        if args.lmcache_mode == "transfer":
            cmd.extend(
                [
                    "--kv-transfer-config",
                    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
                ]
            )
        else:
            cmd.extend(
                [
                    "--kv-offloading-backend",
                    "lmcache",
                    "--kv-offloading-size",
                    str(args.kv_offloading_size),
                    "--disable-hybrid-kv-cache-manager",
                ]
            )
        for extra in args.server_extra_arg:
            cmd.append(extra)
        return cmd, env_updates

    cmd = [
        args.python_executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--enable-lmcache",
    ]
    if args.tp and args.tp > 1:
        cmd.extend(["--tp", str(args.tp)])
    for extra in args.server_extra_arg:
        cmd.append(extra)
    return cmd, env_updates


def load_trace_requests(trace_path: str) -> List[Request]:
    path = Path(trace_path)
    text = path.read_text(encoding="utf-8")
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    else:
        data = json.loads(text)
        if isinstance(data, dict):
            rows = list(data.get("requests") or [])
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"Unsupported trace format in {trace_path}")

    requests: List[Request] = []
    current_arrival = 0.0
    for idx, row in enumerate(rows):
        prompt = row.get("prompt")
        if prompt is None and row.get("messages"):
            prompt = "\n".join(str(msg.get("content", "")) for msg in row["messages"])
        if prompt is None:
            raise ValueError(f"Trace row {idx} is missing both prompt and messages")
        arrival = float(row.get("arrival_time", current_arrival))
        requests.append(Request(request_id=int(row.get("request_id", idx)), prompt=str(prompt), arrival_time=arrival, metadata=row))
        current_arrival = arrival
    return requests
