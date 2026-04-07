#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${PLATFORM_ROOT}/../.." && pwd)"
REPO_ROOT="$(cd "${PLATFORM_ROOT}/.." && pwd)"
ARTIFACT_DIR="${PLATFORM_ROOT}/artifacts/preflight"
OUTPUT_JSON="${ARTIFACT_DIR}/vllm_smoke.json"

HOST_PORT="${HOST_PORT:-18080}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.64}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

mkdir -p "${ARTIFACT_DIR}"

cleanup() {
  docker rm -f nemotron3-super-nvfp4 >/dev/null 2>&1 || true
}
trap cleanup EXIT

pushd "${REPO_ROOT}" >/dev/null
HOST_PORT="${HOST_PORT}" \
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
READY_TIMEOUT="${READY_TIMEOUT}" \
./run_nemotron_nvfp4.sh super
popd >/dev/null

TIMING_FILE="$(mktemp)"
BODY_FILE="$(mktemp)"
curl -sS -o "${BODY_FILE}" \
  -w "%{time_starttransfer}\n%{time_total}\n" \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-smoke' \
  -d '{
    "model": "nemotron-3-super",
    "messages": [{"role":"user","content":"Reply with exactly: smoke-ok"}],
    "temperature": 0,
    "max_tokens": 8
  }' \
  "http://localhost:${HOST_PORT}/v1/chat/completions" > "${TIMING_FILE}"

python3 - <<'PY' "${TIMING_FILE}" "${BODY_FILE}" "${OUTPUT_JSON}" "${HOST_PORT}" "${ENABLE_PREFIX_CACHING}"
import json
import sys
from pathlib import Path

timing_file, body_file, output_file, port, prefix_caching = sys.argv[1:]
timings = Path(timing_file).read_text(encoding="utf-8").strip().splitlines()
body = json.loads(Path(body_file).read_text(encoding="utf-8"))
report = {
    "port": int(port),
    "enable_prefix_caching": prefix_caching == "1",
    "ttft_seconds": float(timings[0]),
    "total_seconds": float(timings[1]),
    "response": body,
}
Path(output_file).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
