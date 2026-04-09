#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${ROOT_DIR}/artifacts/benchmarks"
OUT_TXT="${OUT_DIR}/phase05_transport_contract_${TIMESTAMP}.txt"
BIN="/tmp/phase05_transport_contract_dump"

mkdir -p "${OUT_DIR}"

nvcc -std=c++20 \
  -I"${ROOT_DIR}/third_party/TensorRT-LLM/cpp" \
  -I"${ROOT_DIR}/.venv-trtllm/lib/python3.12/site-packages/flashinfer/data/cutlass/include" \
  -I"${ROOT_DIR}/third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/sm120_blockwise_gemm" \
  "${ROOT_DIR}/proj-2026-04-05-1704/dump_unified_routed_transport_contract.cu" \
  -o "${BIN}"

"${BIN}" | tee "${OUT_TXT}"

echo "transport_contract=${OUT_TXT}"
