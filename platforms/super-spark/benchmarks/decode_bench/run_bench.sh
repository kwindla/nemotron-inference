#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${PLATFORM_ROOT}/../.." && pwd)"
ARTIFACT_DIR="${PLATFORM_ROOT}/artifacts/benchmarks"
MANIFEST_PATH="${1:-${NEMOTRON_FORWARD_MANIFEST:-${PLATFORM_ROOT}/artifacts/manifests/forward_runtime_manifest_unverified.json}}"
FIXTURE_ROOT="${2:-${PLATFORM_ROOT}/testing/oracle/full_model_single_token_short_chat_cuda_v3}"
if [[ $# -ge 1 ]]; then
  shift
fi
if [[ $# -ge 1 ]]; then
  shift
fi

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

CUDA_TAG="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/cuda\1\2/p' | head -n1)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PREFIX="${ARTIFACT_DIR}/single_token_decode_${TIMESTAMP}_${CUDA_TAG:-cuda_unknown}"
JSON_OUT="${OUTPUT_PREFIX}.json"
STDOUT_OUT="${OUTPUT_PREFIX}.stdout.txt"
ENV_OUT="${OUTPUT_PREFIX}.env.txt"

mkdir -p "${ARTIFACT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=${PLATFORM_ROOT}"
  echo "manifest=${MANIFEST_PATH}"
  echo "fixture_root=${FIXTURE_ROOT}"
  echo "binary=${REPO_ROOT}/build/platforms/super-spark/benchmarks/decode_bench/single_token_decode_bench"
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
} > "${ENV_OUT}"

"${REPO_ROOT}/build/platforms/super-spark/benchmarks/decode_bench/single_token_decode_bench" \
  --manifest "${MANIFEST_PATH}" \
  --fixture-root "${FIXTURE_ROOT}" \
  --json-output "${JSON_OUT}" \
  "$@" | tee "${STDOUT_OUT}"
