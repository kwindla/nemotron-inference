#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_PYTHON="${REPO_ROOT}/vllm-env/bin/python"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "run-vllm.sh: vllm-env not found at ${REPO_ROOT}/vllm-env" >&2
  echo "  Build it with: TORCH_CUDA_ARCH_LIST=\"12.0\" MAX_JOBS=6 uv pip install --python vllm-env/bin/python -e third_party/vllm --torch-backend=auto" >&2
  exit 1
fi

# SM120 (RTX 5090) environment:
# - Do NOT set VLLM_FLASH_ATTN_VERSION (no-op in v0.19.0, FA2 is default on SM120)
# - Do NOT set VLLM_USE_FLASHINFER_MOE_FP4 (FlashInfer NVFP4 MoE is broken on SM120)

exec "${VLLM_PYTHON}" "$@"
