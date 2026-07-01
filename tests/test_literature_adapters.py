from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_accurate_baselines.adapter_lib import (
    ExternalCallResult,
    ExternalAdmissionController,
    build_lmcache_command,
    build_sglang_radix_attention_command,
    build_sglang_hicache_command,
    build_vllm_apc_command,
    load_trace_requests,
    vllm_compat_env_updates,
)


def test_build_sglang_hicache_command_contains_official_flags() -> None:
    args = argparse.Namespace(
        python_executable="python",
        model="gpt2",
        host="127.0.0.1",
        port=30000,
        page_size=64,
        hicache_ratio=2.0,
        hicache_size=0.0,
        hicache_io_backend="kernel",
        hicache_write_policy="write_through",
        hicache_mem_layout="page_first",
        hicache_storage_backend="hf3fs",
        hicache_storage_prefetch_policy="wait_complete",
        hicache_storage_backend_extra_config='{"tp_lcm_size":8}',
        tp=2,
        server_extra_arg=["--mem-fraction-static", "0.85"],
    )
    command = build_sglang_hicache_command(args)
    assert "--enable-hierarchical-cache" in command
    assert "--hicache-storage-backend" in command
    assert "hf3fs" in command
    assert "--tp" in command
    assert "2" in command


def test_build_lmcache_command_for_vllm_transfer_mode() -> None:
    args = argparse.Namespace(
        engine="vllm",
        model="gpt2",
        host="127.0.0.1",
        port=8000,
        lmcache_mode="transfer",
        lmcache_chunk_size=256,
        lmcache_config_file=None,
        kv_offloading_size=10.0,
        python_executable="python",
        tp=1,
        server_extra_arg=[],
    )
    command, env = build_lmcache_command(args)
    assert command[:3] == ["vllm", "serve", "gpt2"]
    assert "--kv-transfer-config" in command
    assert env["LMCACHE_CHUNK_SIZE"] == "256"


def test_build_vllm_apc_command_enables_prefix_caching() -> None:
    args = argparse.Namespace(
        model="gpt2",
        host="127.0.0.1",
        port=8000,
        dtype="auto",
        tp=1,
        server_extra_arg=[],
    )
    command = build_vllm_apc_command(args)
    assert command[:3] == ["vllm", "serve", "gpt2"]
    assert "--enable-prefix-caching" in command


def test_vllm_compat_env_defaults_to_v0(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    assert vllm_compat_env_updates()["VLLM_USE_V1"] == "0"
    monkeypatch.setenv("VLLM_USE_V1", "1")
    assert vllm_compat_env_updates()["VLLM_USE_V1"] == "0"


def test_external_admission_write_through_stores_positive_shared_prefix() -> None:
    controller = ExternalAdmissionController("gpt2", runtime="vllm")
    metadata = {"shared_prefix_hint_tokens": 8, "prompt_mode": "templated"}
    tokens = tuple(range(12))
    plan, planned_tokens = controller.plan(" ".join(str(x) for x in tokens), metadata=metadata)

    stored = controller.record_after_request(
        planned_tokens,
        plan,
        metadata=metadata,
        result=ExternalCallResult(request_id=0, latency_ms=120.0, prompt_tokens=12, total_tokens=13),
        allow_bypass_store=True,
    )

    assert stored is True
    assert controller.bank.peek_match(planned_tokens) is not None


def test_external_admission_calibrates_from_runtime_usage() -> None:
    controller = ExternalAdmissionController("gpt2", runtime="vllm")
    original = controller.full_ms_per_token
    tokens = tuple(range(10))
    plan, _ = controller.plan(" ".join(str(x) for x in tokens), metadata={})

    controller.record_after_request(
        tokens,
        plan,
        result=ExternalCallResult(request_id=0, latency_ms=100.0, prompt_tokens=10, total_tokens=11),
    )

    assert controller.full_ms_per_token > original


def test_build_sglang_radix_attention_command_rejects_disabled_radix_cache() -> None:
    args = argparse.Namespace(
        python_executable="python",
        model="gpt2",
        host="127.0.0.1",
        port=30000,
        tp=1,
        attention_backend=None,
        server_extra_arg=["--disable-radix-cache"],
    )
    try:
        build_sglang_radix_attention_command(args)
    except ValueError as exc:
        assert "must not pass --disable-radix-cache" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_trace_requests_supports_json_and_messages() -> None:
    trace_path = Path(__file__).with_name("_trace_requests_fixture.json")
    try:
        trace_path.write_text(
            json.dumps(
                {
                    "requests": [
                        {
                            "request_id": 7,
                            "messages": [{"role": "user", "content": "hello world"}],
                            "arrival_time": 0.25,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        requests = load_trace_requests(str(trace_path))
        assert len(requests) == 1
        assert requests[0].request_id == 7
        assert "hello world" in requests[0].prompt
        assert requests[0].arrival_time == 0.25
    finally:
        trace_path.unlink(missing_ok=True)
