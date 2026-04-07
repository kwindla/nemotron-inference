#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/manifest.json [repeats]" >&2
  exit 2
fi

MANIFEST_PATH="$1"
REPEATS="${2:-3}"
ARTIFACT_DIR="artifacts/benchmarks"
OUTPUT_PREFIX="${ARTIFACT_DIR}/gb10_loader_modes_$(date -u +%Y%m%dT%H%M%SZ)"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

mkdir -p "${ARTIFACT_DIR}"

{
  echo "manifest=${MANIFEST_PATH}"
  echo "repeats=${REPEATS}"
  echo "cwd=$(pwd)"
  echo
  echo "== nvcc --version =="
  nvcc --version || true
  echo
  echo "== nvidia-smi =="
  nvidia-smi || true
  echo
  echo "== cuda installs =="
  ls -d /usr/local/cuda* 2>/dev/null || true
} > "${OUTPUT_PREFIX}.env.txt"

./build/benchmarks/gb10_loader_modes/gb10_loader_modes_bench \
  --manifest "${MANIFEST_PATH}" \
  --repeats "${REPEATS}" \
  --json-output "${OUTPUT_PREFIX}.json" \
  | tee "${OUTPUT_PREFIX}.stdout.txt"

echo "wrote:"
echo "  ${OUTPUT_PREFIX}.json"
echo "  ${OUTPUT_PREFIX}.stdout.txt"
echo "  ${OUTPUT_PREFIX}.env.txt"
