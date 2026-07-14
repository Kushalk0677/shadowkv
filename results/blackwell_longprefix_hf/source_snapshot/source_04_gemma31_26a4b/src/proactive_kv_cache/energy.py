from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EnergySnapshot:
    """Single low-overhead energy/power snapshot.

    gpu_energy_mj is an accumulated counter when the device exposes it. Some
    NVIDIA drivers/GPUs do not expose total energy; callers can then fall back
    to sampled power integration outside this class.
    """

    timestamp_s: float
    gpu_energy_mj: Optional[int] = None
    gpu_power_w: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    source: str = "unavailable"
    error: Optional[str] = None


class NvidiaEnergyMeter:
    """NVML-first GPU energy meter with a small nvidia-smi fallback.

    Primary path uses pynvml total-energy deltas because accumulated energy is
    cleaner than sparse power samples for batch benchmarks. If total energy is
    unavailable, snapshots still include instantaneous power when possible so
    callers can report that energy is unavailable rather than silently faking it.
    """

    def __init__(self, device_index: int = 0, *, enable_smi_fallback: bool = True):
        self.device_index = int(device_index)
        self.enable_smi_fallback = bool(enable_smi_fallback)
        self.available = False
        self.source = "unavailable"
        self.error: Optional[str] = None
        self._pynvml: Any = None
        self._handle: Any = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.available = True
            self.source = "nvml"
        except Exception as exc:  # pragma: no cover - environment dependent
            self.error = str(exc)
            self.available = False
            self.source = "nvidia_smi" if self.enable_smi_fallback else "unavailable"

    def snapshot(self) -> EnergySnapshot:
        if self.available and self._pynvml is not None and self._handle is not None:
            return self._nvml_snapshot()
        if self.enable_smi_fallback:
            return self._smi_snapshot()
        return EnergySnapshot(timestamp_s=time.perf_counter(), source="unavailable", error=self.error)

    def _nvml_snapshot(self) -> EnergySnapshot:
        energy_mj: Optional[int] = None
        power_w: Optional[float] = None
        memory_mb: Optional[float] = None
        err: Optional[str] = None
        try:
            energy_mj = int(self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception as exc:
            err = str(exc)
        try:
            power_w = float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        except Exception:
            pass
        try:
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            memory_mb = float(mem.used) / (1024.0 * 1024.0)
        except Exception:
            pass
        return EnergySnapshot(
            timestamp_s=time.perf_counter(),
            gpu_energy_mj=energy_mj,
            gpu_power_w=power_w,
            gpu_memory_used_mb=memory_mb,
            source="nvml",
            error=err,
        )

    def _smi_snapshot(self) -> EnergySnapshot:
        query = "power.draw,memory.used"
        cmd = [
            "nvidia-smi",
            f"--id={self.device_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2.0)
            first = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            power_w = float(parts[0]) if parts and parts[0] not in {"", "[N/A]"} else None
            memory_mb = float(parts[1]) if len(parts) > 1 and parts[1] not in {"", "[N/A]"} else None
            return EnergySnapshot(
                timestamp_s=time.perf_counter(),
                gpu_power_w=power_w,
                gpu_memory_used_mb=memory_mb,
                source="nvidia_smi",
                error=self.error,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            return EnergySnapshot(timestamp_s=time.perf_counter(), source="unavailable", error=str(exc))

    @staticmethod
    def delta(before: EnergySnapshot, after: EnergySnapshot) -> dict[str, Any]:
        elapsed_s = max(float(after.timestamp_s - before.timestamp_s), 0.0)
        energy_j: Optional[float] = None
        if before.gpu_energy_mj is not None and after.gpu_energy_mj is not None:
            raw_delta_mj = int(after.gpu_energy_mj) - int(before.gpu_energy_mj)
            if raw_delta_mj >= 0:
                energy_j = raw_delta_mj / 1000.0
        return {
            "energy_source": after.source or before.source,
            "energy_elapsed_s": elapsed_s,
            "gpu_energy_j": energy_j,
            "avg_power_w_from_energy": None if energy_j is None or elapsed_s <= 0 else energy_j / elapsed_s,
            "start_gpu_power_w": before.gpu_power_w,
            "end_gpu_power_w": after.gpu_power_w,
            "start_gpu_memory_mb": before.gpu_memory_used_mb,
            "end_gpu_memory_mb": after.gpu_memory_used_mb,
            "energy_error": after.error or before.error,
        }


def measure_idle_baseline(meter: NvidiaEnergyMeter, duration_s: float) -> dict[str, Any]:
    """Measure idle GPU energy over a short window.

    Keep this outside measured engine execution. The result can be normalized to
    an idle watt estimate and subtracted from benchmark energy windows.
    """

    duration_s = max(float(duration_s), 0.0)
    before = meter.snapshot()
    if duration_s > 0:
        time.sleep(duration_s)
    after = meter.snapshot()
    delta = NvidiaEnergyMeter.delta(before, after)
    energy_j = delta.get("gpu_energy_j")
    elapsed_s = float(delta.get("energy_elapsed_s") or 0.0)
    delta["idle_baseline_duration_s"] = duration_s
    delta["idle_baseline_power_w"] = None if energy_j is None or elapsed_s <= 0 else float(energy_j) / elapsed_s
    return delta
