#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_JSON="${SCRIPT_DIR}/vllm_baseline_results.json"

export PYTHONPATH="${REPO_ROOT}/third_party/vllm${PYTHONPATH:+:${PYTHONPATH}}"

# Force the Nemotron Nano NVFP4 path onto FlashInfer CUTLASS rather than TRTLLM.
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass

# vLLM v0.19.0 does not consume VLLM_MOE_PADDING directly on CUDA, but some
# environments export it when steering FlashInfer FP4 kernels. Keep it explicit.
export VLLM_MOE_PADDING=0

python3 "${SCRIPT_DIR}/bench_vllm_nano.py" --output "${OUTPUT_JSON}" "$@"
