#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_JSON="${SCRIPT_DIR}/vllm_baseline_results.json"
DEFAULT_PYTHON_BIN="${REPO_ROOT}/vllm-env-cu128/bin/python"
PYTHON_BIN="${VLLM_PYTHON:-${DEFAULT_PYTHON_BIN}}"
export PYTHONPATH="${REPO_ROOT}/third_party/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/bench_vllm_nano.py" \
  --output "${OUTPUT_JSON}" \
  --trust-remote-code \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
  --moe-backend flashinfer_cutlass \
  "$@"
