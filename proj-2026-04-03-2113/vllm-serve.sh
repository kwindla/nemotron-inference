#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_PYTHON="${REPO_ROOT}/vllm-env/bin/python"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "vllm-serve.sh: vllm-env not found at ${REPO_ROOT}/vllm-env" >&2
  exit 1
fi

MODEL="${VLLM_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"

exec "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --dtype auto \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  "$@"
