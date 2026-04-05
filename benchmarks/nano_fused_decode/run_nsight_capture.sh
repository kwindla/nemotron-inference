#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/profiles"

usage() {
  cat >&2 <<'EOF'
Usage: NEMOTRON_FORWARD_MANIFEST=/path/to/manifest.json run_nsight_capture.sh [-- benchmark-args]
   or: run_nsight_capture.sh /path/to/manifest.json [-- benchmark-args]

Runs nano_fused_decode_bench under `nsys profile` with `--mode=profile-ready`.
Existing `NEMOTRON_*`, `CUDA_VISIBLE_DEVICES`, and `LD_LIBRARY_PATH` environment
variables are passed through unchanged.
EOF
}

MANIFEST_PATH="${NEMOTRON_FORWARD_MANIFEST:-${1:-}}"
if [[ -z "${MANIFEST_PATH}" ]]; then
  usage
  exit 1
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
  shift
fi
if [[ $# -gt 0 && "${1}" == "--" ]]; then
  shift
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "run_nsight_capture.sh: nsys not found in PATH" >&2
  exit 1
fi

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

BUILD_DIR="${NEMOTRON_BUILD_DIR:-}"
if [[ -z "${BUILD_DIR}" ]]; then
  for candidate in build-sm120-relwithdebinfo; do
    if [[ -x "${ROOT_DIR}/${candidate}/benchmarks/nano_fused_decode/nano_fused_decode_bench" ]]; then
      BUILD_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${BUILD_DIR}" ]]; then
  echo "run_nsight_capture.sh: nano_fused_decode_bench not found; set NEMOTRON_BUILD_DIR or build the benchmark first" >&2
  exit 1
fi

BINARY="${ROOT_DIR}/${BUILD_DIR}/benchmarks/nano_fused_decode/nano_fused_decode_bench"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PREFIX="${ARTIFACT_DIR}/nano_fused_decode_profile_ready_${TIMESTAMP}"
STDOUT_OUT="${OUTPUT_PREFIX}.stdout.txt"
ENV_OUT="${OUTPUT_PREFIX}.env.txt"

mkdir -p "${ARTIFACT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=${ROOT_DIR}"
  echo "build_dir=${BUILD_DIR}"
  echo "binary=${BINARY}"
  echo "manifest=${MANIFEST_PATH}"
  echo "uname=$(uname -a)"
  echo
  echo "[forward-env]"
  env | grep -E '^(NEMOTRON_|CUDA_VISIBLE_DEVICES=|LD_LIBRARY_PATH=)' | sort || true
  echo
  echo "[nsys]"
  nsys --version || true
} > "${ENV_OUT}"

export NEMOTRON_FORWARD_MANIFEST="${MANIFEST_PATH}"

nsys profile \
  --force-overwrite true \
  --output "${OUTPUT_PREFIX}" \
  "${BINARY}" \
  --manifest "${MANIFEST_PATH}" \
  --mode=profile-ready \
  "$@" 2>&1 | tee "${STDOUT_OUT}"
