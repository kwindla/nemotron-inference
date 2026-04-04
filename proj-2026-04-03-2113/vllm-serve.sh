#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_VLLM_PYTHON="${REPO_ROOT}/vllm-env-cu128/bin/python"
VLLM_PYTHON="${VLLM_PYTHON:-${DEFAULT_VLLM_PYTHON}}"
export PYTHONPATH="${REPO_ROOT}/third_party/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "vllm-serve.sh: python not found at ${VLLM_PYTHON}" >&2
  exit 1
fi

MODEL="${VLLM_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

exec "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --dtype auto \
  --trust-remote-code \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  --moe-backend flashinfer_cutlass \
  "$@"
