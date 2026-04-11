#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/benchmarks"
MANIFEST_PATH="${NEMOTRON_FORWARD_MANIFEST:-${ROOT_DIR}/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json}"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

BUILD_DIR="${NEMOTRON_BUILD_DIR:-build-sm120-relwithdebinfo}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_TAG="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/cuda\1\2/p' | head -n1)"
OUTPUT_PREFIX="${ARTIFACT_DIR}/nano_shared_prefill_${TIMESTAMP}_${CUDA_TAG:-cuda_unknown}"
JSON_OUT="${OUTPUT_PREFIX}.json"
STDOUT_OUT="${OUTPUT_PREFIX}.stdout.txt"
ENV_OUT="${OUTPUT_PREFIX}.env.txt"
RAW_DIR="${OUTPUT_PREFIX}_raw"

mkdir -p "${ARTIFACT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=${ROOT_DIR}"
  echo "build_dir=${BUILD_DIR}"
  echo "manifest=${MANIFEST_PATH}"
  echo "binary=${ROOT_DIR}/${BUILD_DIR}/benchmarks/nano_fused_decode/nano_fused_decode_bench"
  echo "uname=$(uname -a)"
  echo
  echo "[forward-env]"
  env | grep -E '^(NEMOTRON_|CUDA_VISIBLE_DEVICES=|LD_LIBRARY_PATH=)' | sort || true
  echo
  echo "[nvcc]"
  nvcc --version || true
  echo
  echo "[nvidia-smi]"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi
} > "${ENV_OUT}"

python3 "${ROOT_DIR}/tools/benchmark_analysis/run_shared_prefill_benchmark.py" \
  --build-dir "${BUILD_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --json-output "${JSON_OUT}" \
  --raw-dir "${RAW_DIR}" \
  "$@" | tee "${STDOUT_OUT}"
