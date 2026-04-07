#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_PYTHON="${VLLM_PYTHON:-${REPO_ROOT}/vllm-env-cu128/bin/python}"
MODEL_PATH="${VLLM_MODEL_PATH:-${REPO_ROOT}/artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8011}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-nemotron-nano-vllm}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-0}"

export PYTHONPATH="${REPO_ROOT}/third_party/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "vllm python not found at ${VLLM_PYTHON}" >&2
  exit 1
fi

args=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --dtype auto
  --trust-remote-code
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --enforce-eager
  --moe-backend flashinfer_cutlass
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  args+=(--enable-prefix-caching)
fi

exec "${VLLM_PYTHON}" "${args[@]}" "$@"
