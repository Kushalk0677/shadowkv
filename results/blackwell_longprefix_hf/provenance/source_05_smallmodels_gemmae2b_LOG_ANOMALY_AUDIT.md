# Log Anomaly Audit

- Measured benchmark failures: none (`failed=0/180`, `failed=0/30`).
- Smoke failures: none (`failed=0/18`, `failed=0/3`).
- CUDA OOM, CUDA error, Python traceback, killed process, or external-workload abort: none.
- Expected preflight warning: ComfyUI queue endpoint on port 8188 was unavailable because ComfyUI was not running. No ComfyUI workload was active.
- Known orchestration event: the six-model wrapper reported an unexpected EOF after all 180 cells and aggregate files completed because its script file was replaced while still executing. Production restoration succeeded and the event did not occur inside a measured cell.
- Gemma 4 E2B used a separate wrapper and completed normally.
