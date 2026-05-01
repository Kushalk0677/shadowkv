from __future__ import annotations

import threading
import time
from typing import Callable

from .config_loader import CONFIG, RuntimeConfig


class ConfigWatcher:
    def __init__(self, config: RuntimeConfig = CONFIG, interval_s: float = 1.0, on_reload: Callable[[RuntimeConfig], None] | None = None) -> None:
        self.config = config
        self.interval_s = max(float(interval_s), 0.1)
        self.on_reload = on_reload
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_hash = config.file_hash

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name='shadowkv-config-watcher', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.config.reload_if_changed()
            current_hash = self.config.file_hash
            if current_hash != self._last_hash:
                self._last_hash = current_hash
                if self.on_reload:
                    self.on_reload(self.config)
