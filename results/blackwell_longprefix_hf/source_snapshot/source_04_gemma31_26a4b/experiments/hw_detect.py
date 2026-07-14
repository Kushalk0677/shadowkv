from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proactive_kv_cache.config_loader import CONFIG, RuntimeConfig


def _nvml_init():
    try:
        import pynvml
        pynvml.nvmlInit()
        device_index = 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return pynvml, handle
    except Exception:
        return None


def _detect_gpu_memory_mb():
    """Detect total GPU memory in MB via torch or pynvml."""
    # Try pynvml first (most reliable)
    state = _nvml_init()
    if state is not None:
        try:
            pynvml, handle = state
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.total / (1024 * 1024))
        except Exception:
            pass
    # Fallback: torch
    try:
        import torch
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return int(prop.total_mem / (1024 * 1024))
    except Exception:
        pass
    return None


def _detect_gpu_name():
    """Detect GPU name via pynvml or torch."""
    state = _nvml_init()
    if state is not None:
        try:
            pynvml, handle = state
            return pynvml.nvmlDeviceGetName(handle)
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _detect_pcie_bandwidth_gbps():
    """Estimate PCIe bandwidth in GB/s.

    Tries nvidia-smi first for PCIe link info, falls back to
    GPU-family heuristics, then pure default.
    """
    # Method 1: nvidia-smi --query-gpu=pci.link.width,pci.link.gen
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.link.width,pci.link.gen", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if proc.returncode == 0:
            line = proc.stdout.strip().split("\n")[0].strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                lane_width = int(parts[0])
                pcie_gen_str = parts[1]
                gen_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
                pcie_gen = gen_map.get(pcie_gen_str, 3)
                # GB/s ≈ lanes × (5 GB/s per lane for gen3, 9 for gen4, 16 for gen5)
                per_lane = {1: 0.5, 2: 1.0, 3: 5.0, 4: 9.0, 5: 16.0}
                bandwidth_gbps = float(lane_width * per_lane.get(pcie_gen, 5.0))
                return round(bandwidth_gbps, 1)
    except Exception:
        pass

    # Method 2: pynvml
    state = _nvml_init()
    if state is not None:
        try:
            pynvml, handle = state
            info = pynvml.nvmlDeviceGetPciInfo(handle)
            # lane_count gives actual *used* lanes; gen maps to PCIe version
            lane_count = info.lane_count
            # PCIE_BUS_ID etc are available but bandwidth calc is same
            # Need to find generation from device capabilities
            # pynvml doesn't always expose gen, so fall through
            if lane_count > 0:
                pcie_gen = 3  # reasonable default
                # Try to detect gen from PCIe speed info if available
                # Not directly available in most pynvml builds
                per_lane = {3: 5.0, 4: 9.0, 5: 16.0}
                bandwidth_gbps = float(lane_count * per_lane.get(pcie_gen, 5.0))
                return round(bandwidth_gbps, 1)
        except Exception:
            pass

    # Method 3: GPU family heuristics
    gpu_name = _detect_gpu_name()
    if gpu_name:
        gpu_name_lower = str(gpu_name).lower()
        # A100/H100 usually have PCIe gen4, gen5
        if "h100" in gpu_name_lower:
            return 64.0  # 8 lanes x gen4 typical
        if "a100" in gpu_name_lower:
            return 64.0
        if "v100" in gpu_name_lower:
            return 32.0  # PCIe gen3
        # Consumer GPUs: typically 8 or 16 lanes gen3-4
        return 32.0  # 8 lanes gen4 conservative

    # Method 4: pure default
    return 15.0


def _detect_cpu_memory_mb():
    """Detect total system RAM in MB."""
    # Linux
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                # Value in kB
                kb = int(line.split()[1])
                return kb // 1024
    except Exception:
        pass

    # Try nproc/fallback
    try:
        import subprocess
        proc = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=2)
        if proc.returncode == 0:
            lines = proc.stdout.strip().split("\n")
            for line in lines:
                if "Mem:" in line:
                    parts = line.split()
                    return int(parts[1])
    except Exception:
        pass

    # Windows
    try:
        import subprocess
        proc = subprocess.run(["wmic", "OS", "get", "TotalVisibleMemorySize"], capture_output=True, text=True, timeout=3)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.isdigit() and int(line) > 0:
                    # KB
                    return int(line) // 1024
    except Exception:
        pass

    # Fallback: assume system RAM
    return 16384


def _detect_device():
    """Detect available device (cuda or cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def detect_hardware(config_path: str | None = None) -> dict[str, float | None]:
    """Detect hardware capabilities and return dict of config values.

    Returns detected values keyed by config path. Missing detections
    return None and will be skipped (not written to config).
    """
    config = RuntimeConfig(config_path) if config_path else CONFIG
    device = _detect_device()

    detected = {}

    # GPU detection (only if cuda available)
    if device == "cuda":
        gpu_mem = _detect_gpu_memory_mb()
        if gpu_mem is not None:
            detected["hardware.max_gpu_memory_mb"] = gpu_mem
        pcie_bw = _detect_pcie_bandwidth_gbps()
        detected["hardware.pcie_bandwidth_gbps"] = pcie_bw
    else:
        cpu_mem = _detect_cpu_memory_mb()
        if cpu_mem is not None:
            detected["hardware.max_cpu_memory_mb"] = cpu_mem

    return detected


def apply_detected_config(config_path: str | None = None, log: bool = True):
    """Run detection and write values into config.yaml.

    Only writes non-None detected values. Preserves existing config keys.
    Returns dict of all attempted detections.
    """
    config = RuntimeConfig(config_path) if config_path else CONFIG
    detected = detect_hardware(config_path)

    if not detected:
        if log:
            print("[hw_detect] No GPU detected — skipping hardware detection.")
        return detected

    config.update({k: v for k, v in detected.items() if v is not None})
    config.write(config.path)

    if log:
        lines = [f"[hw_detect] Detected hardware values for device={_detect_device() or 'cpu'}:"]
        for key, value in detected.items():
            status = f"{value}" if value is not None else "(skipped)"
            lines.append(f"    {key}: {status}")
        print("\n".join(lines))
        if config.path.exists():
            print(f"[hw_detect] Written to {config.path}")

    return detected


if __name__ == "__main__":
    detected = apply_detected_config(log=True)
    for k, v in detected.items():
        print(f"  {k}: {v}")
