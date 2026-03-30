#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/benchmarks"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

CUDA_TAG="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/cuda\1\2/p' | head -n1)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PREFIX="${ARTIFACT_DIR}/gb10_dense_gemm_default_${TIMESTAMP}_${CUDA_TAG:-cuda_unknown}"
JSON_OUT="${OUTPUT_PREFIX}.json"
STDOUT_OUT="${OUTPUT_PREFIX}.stdout.txt"
ENV_OUT="${OUTPUT_PREFIX}.env.txt"

mkdir -p "${ARTIFACT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=${ROOT_DIR}"
  echo "binary=${ROOT_DIR}/build/benchmarks/gb10_dense_gemm/gb10_dense_gemm_bench"
  echo "uname=$(uname -a)"
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
  echo
  echo "[cuda-install-dirs]"
  ls -d /usr/local/cuda* 2>/dev/null || true
} > "${ENV_OUT}"

"${ROOT_DIR}/build/benchmarks/gb10_dense_gemm/gb10_dense_gemm_bench" \
  --json-output "${JSON_OUT}" \
  "$@" | tee "${STDOUT_OUT}"
