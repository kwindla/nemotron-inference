#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_VLLM_PYTHON="${REPO_ROOT}/vllm-env-cu128/bin/python"
VLLM_PYTHON="${VLLM_PYTHON:-${DEFAULT_VLLM_PYTHON}}"
export PYTHONPATH="${REPO_ROOT}/third_party/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "run-vllm.sh: python not found at ${VLLM_PYTHON}" >&2
  echo "  Expected working env: ${REPO_ROOT}/vllm-env-cu128/bin/python" >&2
  exit 1
fi

# Local RTX 5090 path:
# - Use the cu128-based Python env. The cu130 env trips lower-level GEMM failures.
# - Keep PYTHONPATH pointed at the local third_party/vllm checkout.

exec "${VLLM_PYTHON}" "$@"
