#!/usr/bin/env bash
set -uo pipefail

PROJECT=/home/jade_hand/research/shadowkv/hf_blackwell_semantic_n128_longprefix_20260710
RUN_ID=gemma4_31b_26b_a4b_longprefix_n128_clean_20260711
SMOKE_ID=gemma4_31b_26b_a4b_longprefix_smoke_20260711
RESULTS="$PROJECT/results_blackwell_semantic_${RUN_ID}"
SMOKE_RESULTS="$PROJECT/results_smoke_${SMOKE_ID}"
RUN_LOGS="$PROJECT/run_logs"
LOG="$RUN_LOGS/${RUN_ID}.log"
STATUS="$RUN_LOGS/${RUN_ID}.status"
NOHUP_LOG="$RUN_LOGS/${RUN_ID}.nohup.log"
CONTAINER=shadowkv-hf-${RUN_ID//_/-}
IMAGE=shadowkv-hf-blackwell:20260706
PRODUCTION_ROOT=/home/jade_hand/active/services/qwen36-27b-fp8-vllm
ABORT_MARKER="$RUN_LOGS/${RUN_ID}.external_gpu_abort"
GUARD_PID=

mkdir -p "$RUN_LOGS" "$RESULTS/metadata"
rm -f "$ABORT_MARKER"
exec > >(tee -a "$LOG") 2>&1

write_status() {
  printf '%s\n' "$1" > "$STATUS"
}

restore_production() {
  if [[ -n "${GUARD_PID:-}" ]]; then
    kill "$GUARD_PID" >/dev/null 2>&1 || true
    wait "$GUARD_PID" 2>/dev/null || true
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  sudo -n chown -R jade_hand:users "$RESULTS" "$SMOKE_RESULTS" "$RUN_LOGS" 2>/dev/null || true
  cd "$PRODUCTION_ROOT" || return 0
  docker compose up -d
  for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:8014/v1/models > "$RESULTS/metadata/production_models_after.json" 2>/dev/null; then
      printf '%s\n' 'production_restored=true' >> "$STATUS"
      return 0
    fi
    sleep 5
  done
  printf '%s\n' 'production_restored=false' >> "$STATUS"
}

gpu_guard() {
  while kill -0 "$BASHPID" 2>/dev/null; do
    if pgrep -f '[L]iftoff.*x86_64' >/dev/null 2>&1; then
      date -Is > "$ABORT_MARKER"
      printf '%s\n' 'reason=Liftoff detected during benchmark' >> "$ABORT_MARKER"
      docker stop -t 10 "$CONTAINER" >/dev/null 2>&1 || true
      return
    fi
    queue_json=$(curl -fsS http://127.0.0.1:8188/queue 2>/dev/null || printf '%s' '{}')
    if printf '%s' "$queue_json" | jq -e '(.queue_running | length) > 0 or (.queue_pending | length) > 0' >/dev/null 2>&1; then
      date -Is > "$ABORT_MARKER"
      printf '%s\n' 'reason=ComfyUI queue became active during benchmark' >> "$ABORT_MARKER"
      docker stop -t 10 "$CONTAINER" >/dev/null 2>&1 || true
      return
    fi
    sleep 5
  done
}

run_container() {
  docker run --rm --name "$CONTAINER" \
    --device nvidia.com/gpu=all \
    --ipc=host \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v /datapool/cache/huggingface:/root/.cache/huggingface \
    -v "$PROJECT":/workspace \
    -w /workspace \
    --entrypoint /usr/bin/python3 \
    "$IMAGE" \
    blackwell_semantic_n128/run_blackwell_semantic_n128.py "$@"
}

trap restore_production EXIT

write_status "PREPARING $(date -Is)"
printf 'run_id=%s\nproject=%s\nresults=%s\nimage=%s\n' "$RUN_ID" "$PROJECT" "$RESULTS" "$IMAGE" > "$RESULTS/metadata/run_identity.txt"
uname -a > "$RESULTS/metadata/uname.txt"
lscpu > "$RESULTS/metadata/lscpu.txt"
nvidia-smi -q > "$RESULTS/metadata/nvidia_smi_q_before.txt"
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv > "$RESULTS/metadata/gpu_baseline_before.csv"
docker image inspect "$IMAGE" > "$RESULTS/metadata/docker_image_inspect.json"
docker run --rm --entrypoint /usr/bin/python3 "$IMAGE" -m pip freeze > "$RESULTS/metadata/runtime_versions.txt"
sha256sum \
  "$PROJECT/blackwell_semantic_n128/run_blackwell_semantic_n128.py" \
  "$PROJECT/experiments/run_benchmark.py" \
  "$PROJECT/src/proactive_kv_cache/datasets.py" \
  "$PROJECT/src/proactive_kv_cache/engines.py" \
  "$PROJECT/src/proactive_kv_cache/models.py" \
  "$PROJECT/tests/test_backend_regressions.py" \
  "$PROJECT/tests/test_overhaul_features.py" \
  "$PROJECT/tests/test_shadowkv_plus.py" \
  > "$RESULTS/metadata/source_sha256.txt"
cp "$PROJECT/OPERATOR_NOTES_20260710.md" "$RESULTS/metadata/"

if pgrep -f '[L]iftoff.*x86_64' >/dev/null 2>&1; then
  write_status "BLOCKED_LIFTOFF $(date -Is)"
  exit 20
fi

queue_json=$(curl -fsS http://127.0.0.1:8188/queue 2>/dev/null || printf '%s' '{}')
if printf '%s' "$queue_json" | jq -e '(.queue_running | length) > 0 or (.queue_pending | length) > 0' >/dev/null 2>&1; then
  write_status "BLOCKED_COMFYUI $(date -Is)"
  exit 21
fi
curl -fsS -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' \
  -d '{"unload_models":true,"free_memory":true}' >/dev/null || true
sleep 5

docker stop qwen36-27b-fp8-vllm >/dev/null 2>&1 || true
write_status "SMOKE_RUNNING $(date -Is)"
gpu_guard &
GUARD_PID=$!

run_container \
  --models google/gemma-4-31B-it google/gemma-4-26B-A4B-it \
  --datasets ag_news \
  --seeds 42 \
  --n_requests 16 \
  --semantic_shared_prefix_repeats 4 \
  --semantic_shared_prefix_mode common_scaffold \
  --results_root "$(basename "$SMOKE_RESULTS")"
SMOKE_RC=$?

if [[ "$SMOKE_RC" -ne 0 ]]; then
  write_status "SMOKE_FAILED rc=$SMOKE_RC $(date -Is)"
  exit "$SMOKE_RC"
fi

python3 - "$SMOKE_RESULTS" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / "reuse_path_breakdown.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
plus = [row for row in rows if row["engine"] == "shadow_kv_plus"]
if len(rows) != 4 or len(plus) != 2:
    raise SystemExit(f"unexpected smoke rows: total={len(rows)} plus={len(plus)}")
for row in plus:
    if row["path_reading"] != "exact_scaffold_only":
        raise SystemExit(f"unexpected reuse path for {row['model']}: {row['path_reading']}")
    if int(float(row["reuse_successes"])) != 15:
        raise SystemExit(f"unexpected reuse successes for {row['model']}: {row['reuse_successes']}")
    if float(row.get("hit_rate") or 0) != 0.9375:
        raise SystemExit(f"unexpected hit rate for {row['model']}: {row.get('hit_rate')}")
PY
SMOKE_AUDIT_RC=$?
if [[ "$SMOKE_AUDIT_RC" -ne 0 ]]; then
  write_status "SMOKE_AUDIT_FAILED rc=$SMOKE_AUDIT_RC $(date -Is)"
  exit "$SMOKE_AUDIT_RC"
fi

write_status "RUNNING $(date -Is)"
run_container \
  --models google/gemma-4-31B-it google/gemma-4-26B-A4B-it \
  --datasets ag_news alpaca_eval banking77 cnn_dailymail daily_dialog dolly oasst1 samsum ultrachat xsum \
  --seeds 42 \
  --n_requests 128 \
  --semantic_shared_prefix_repeats 4 \
  --semantic_shared_prefix_mode common_scaffold \
  --results_root "$(basename "$RESULTS")"
RUN_RC=$?

kill "$GUARD_PID" >/dev/null 2>&1 || true
wait "$GUARD_PID" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv > "$RESULTS/metadata/gpu_after.csv"

if [[ -f "$ABORT_MARKER" ]]; then
  write_status "ABORTED_EXTERNAL_GPU_WORKLOAD $(date -Is)"
  exit 22
fi
if [[ "$RUN_RC" -ne 0 ]]; then
  write_status "FAILED rc=$RUN_RC $(date -Is)"
  exit "$RUN_RC"
fi

write_status "COMPLETE $(date -Is)"
exit 0
