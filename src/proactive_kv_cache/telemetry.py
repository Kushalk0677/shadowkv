from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open('a', encoding='utf-8')

    def write(self, row: Mapping[str, Any]) -> None:
        payload = dict(row)
        payload.setdefault('logged_at_s', time.time())
        self._fh.write(json.dumps(payload, sort_keys=True, default=str) + '\n')
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _run_text(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch

        info['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_capability'] = '.'.join(str(x) for x in torch.cuda.get_device_capability(0))
    except Exception as exc:
        info['torch_gpu_probe_error'] = str(exc)
    smi = _run_text(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])
    if smi:
        info['nvidia_smi'] = smi
    return info


def build_run_manifest(*, args: Any, config_snapshot: Mapping[str, Any], config_hash: str | None) -> dict[str, Any]:
    git_commit = _run_text(['git', 'rev-parse', 'HEAD'])
    env_keys = ['CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'SHADOWKV_CONFIG', 'VLLM_USE_MODELSCOPE']
    return {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'git_commit': git_commit,
        'config_hash_sha256': config_hash,
        'config_version': config_snapshot.get('version'),
        'config_profile_id': config_snapshot.get('profile_id'),
        'seed': getattr(args, 'seed', None),
        'model': getattr(args, 'model', None),
        'backend': getattr(args, 'backend', None),
        'device': getattr(args, 'device', None),
        'dtype': getattr(args, 'dtype', None),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'gpu': _gpu_info(),
        'env': {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
        'config_snapshot': dict(config_snapshot),
    }
