from argparse import Namespace
from pathlib import Path

import pytest

from experiments.run_benchmark import EXTERNAL_RUNTIME_BASELINE_NAMES, RUNTIME_BASELINE_NAMES, build_engine, list_engine_names
from literature_accurate_baselines.adapter_lib import ManagedServer, OpenAICompatClient
from literature_accurate_baselines.run_runtime_cache_baseline import _wait_until_ready_or_server_exit
from proactive_kv_cache.engines import AdmissionControlledRuntimeCacheEngine, RuntimeNativeCacheEngine
from proactive_kv_cache.models import FakeBackend


def _args(**overrides):
    values = {
        "backend": "vllm",
        "include_runtime_baselines": True,
        "include_experimental": False,
        "include_semantic_ablations": False,
        "engines": None,
        "max_memory_mb": 16,
    }
    values.update(overrides)
    return Namespace(**values)


def test_runtime_baselines_are_additive_to_default_engine_list():
    names = list_engine_names(_args())
    for name in RUNTIME_BASELINE_NAMES:
        assert name in names
    assert "native_prefix_cache" in names
    assert "shadow_kv_plus_scaffold_only" not in names


def test_runtime_baseline_builds_plain_and_admission_variants():
    backend = FakeBackend()
    plain = build_engine(_args(), backend, "vllm_apc")
    controlled = build_engine(_args(), backend, "vllm_apc_shadowkv_plus")

    assert isinstance(plain, RuntimeNativeCacheEngine)
    assert isinstance(controlled, AdmissionControlledRuntimeCacheEngine)
    assert plain.name == "vllm_apc"
    assert controlled.name == "vllm_apc_shadowkv_plus"
    assert controlled.engine_metrics["admission_controller_enabled"] is True


def test_admission_controlled_runtime_baseline_serves_requests():
    backend = FakeBackend()
    engine = build_engine(_args(), backend, "vllm_apc_shadowkv_plus")
    metadata = {"prompt_mode": "templated", "shared_prefix_hint_tokens": 4}

    first = engine.serve_tokens(1, (1, 2, 3, 4, 5), metadata=metadata)
    second = engine.serve_tokens(2, (1, 2, 3, 4, 6), metadata=metadata)

    assert first.was_cache_hit is False
    assert second.request_id == 2
    assert engine.engine_metrics["admission_plans_total"] == 2


def test_external_runtime_baselines_are_not_in_process_placeholders():
    backend = FakeBackend()
    for name in EXTERNAL_RUNTIME_BASELINE_NAMES:
        with pytest.raises(ValueError, match="external runtime baseline"):
            build_engine(_args(), backend, name)


class ExitedServer:
    def __init__(self, stderr_path: Path) -> None:
        self.stderr_path = stderr_path
        self.stdout_path = stderr_path.with_suffix(".stdout.txt")

    def poll(self):
        return 7

    def tail_logs(self, max_lines: int = 80) -> str:
        server = ManagedServer(
            [],
            cwd=None,
            env_updates=None,
            stdout_path=self.stdout_path,
            stderr_path=self.stderr_path,
        )
        return server.tail_logs(max_lines=max_lines)


def test_wait_until_ready_fails_fast_when_server_exits():
    stderr_path = Path(__file__).with_name("_server_exit_fixture.stderr.txt")
    try:
        stderr_path.write_text("boom\nmissing dependency\n", encoding="utf-8")
        client = OpenAICompatClient("http://127.0.0.1:9", model="gpt2")

        with pytest.raises(RuntimeError) as exc_info:
            _wait_until_ready_or_server_exit(client, ExitedServer(stderr_path), timeout_s=30.0)

        message = str(exc_info.value)
        assert "exited before becoming ready" in message
        assert "missing dependency" in message
    finally:
        stderr_path.unlink(missing_ok=True)
