from __future__ import annotations

import ast
import hashlib
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return Path(os.environ.get('SHADOWKV_CONFIG', _project_root() / 'config' / 'config.yaml'))


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text == '' or text.lower() in {'null', 'none', '~'}:
        return None
    if text.lower() == 'true':
        return True
    if text.lower() == 'false':
        return False
    if (text.startswith('[') and text.endswith(']')) or (text.startswith('{') and text.endswith('}')):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _simple_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        key, sep, value = stripped.partition(':')
        if not sep:
            continue
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value == '':
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _simple_yaml_dump(data: Mapping[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = ' ' * indent
    for key, value in data.items():
        if isinstance(value, Mapping):
            lines.append(f'{pad}{key}:')
            lines.append(_simple_yaml_dump(value, indent + 2).rstrip())
            continue
        if value is None:
            rendered = 'null'
        elif isinstance(value, bool):
            rendered = 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            rendered = str(value)
        elif isinstance(value, (list, tuple, dict)):
            rendered = repr(value)
        else:
            rendered = str(value)
        lines.append(f'{pad}{key}: {rendered}')
    return '\n'.join(lines) + '\n'


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        return dict(loaded)
    except Exception:
        return _simple_yaml_load(text)


def _dump_yaml(data: Mapping[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(dict(data), sort_keys=False)
    except Exception:
        return _simple_yaml_dump(data)


class RuntimeConfig:
    """Small process-wide config singleton with mtime/hash based reloads."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._path = Path(path) if path else _default_config_path()
        self._data: dict[str, Any] = {}
        self._mtime_ns: int | None = None
        self._hash: str | None = None
        self.load(self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def version(self) -> str:
        return str(self.get('version', 'unknown'))

    @property
    def file_hash(self) -> str | None:
        self.reload_if_changed()
        return self._hash

    def load(self, path: str | Path | None = None) -> None:
        with self._lock:
            if path is not None:
                self._path = Path(path)
            if not self._path.exists():
                self._data = {}
                self._mtime_ns = None
                self._hash = None
                return
            payload = self._path.read_bytes()
            self._data = _load_yaml(self._path)
            self._mtime_ns = self._path.stat().st_mtime_ns
            self._hash = hashlib.sha256(payload).hexdigest()

    def reload_if_changed(self, force: bool = False) -> None:
        with self._lock:
            if not self._path.exists():
                return
            mtime_ns = self._path.stat().st_mtime_ns
            if force or self._mtime_ns != mtime_ns:
                self.load(self._path)

    def snapshot(self) -> dict[str, Any]:
        self.reload_if_changed()
        with self._lock:
            return deepcopy(self._data)

    def get(self, dotted_path: str, default: Any = None) -> Any:
        self.reload_if_changed()
        current: Any = self._data
        for part in dotted_path.split('.'):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def update(self, updates: Mapping[str, Any]) -> None:
        with self._lock:
            for dotted_path, value in updates.items():
                target = self._data
                parts = str(dotted_path).split('.')
                for part in parts[:-1]:
                    child = target.get(part)
                    if not isinstance(child, dict):
                        child = {}
                        target[part] = child
                    target = child
                target[parts[-1]] = value

    def write(self, path: str | Path | None = None) -> None:
        with self._lock:
            target = Path(path) if path else self._path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_dump_yaml(self._data), encoding='utf-8')
            self.load(target)


CONFIG = RuntimeConfig()
