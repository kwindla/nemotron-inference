#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-trtllm"
MODEL_PATH="${TRTLLM_MODEL_PATH:-${REPO_ROOT}/artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
CONFIG_PATH="${TRTLLM_CONFIG_PATH:-${SCRIPT_DIR}/trtllm_nano_serve.yaml}"
HOST="${TRTLLM_HOST:-127.0.0.1}"
PORT="${TRTLLM_PORT:-8012}"

if [[ ! -x "${VENV_DIR}/bin/trtllm-serve" ]]; then
  echo "trtllm-serve not found in ${VENV_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/TensorRT-LLM${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${VENV_DIR}/lib:${VENV_DIR}/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TRTLLM_SERVER_DISABLE_GC="${TRTLLM_SERVER_DISABLE_GC:-1}"
export TRTLLM_WORKER_DISABLE_GC="${TRTLLM_WORKER_DISABLE_GC:-1}"

exec "${VENV_DIR}/bin/trtllm-serve" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust_remote_code \
  --config "${CONFIG_PATH}" \
  --served_model_name "${TRTLLM_SERVED_MODEL_NAME:-nemotron-nano-trtllm}"
