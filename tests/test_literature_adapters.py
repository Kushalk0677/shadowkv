from __future__ import annotations

import argparse
import json

from literature_accurate_baselines.adapter_lib import (
    build_lmcache_command,
    build_sglang_radix_attention_command,
    build_sglang_hicache_command,
    build_vllm_apc_command,
    load_trace_requests,
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


def test_load_trace_requests_supports_json_and_messages(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
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
