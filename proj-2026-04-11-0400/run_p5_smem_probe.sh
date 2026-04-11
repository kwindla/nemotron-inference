#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/proj-2026-04-11-0400/p5_smem_probe.cu"
BIN="$ROOT/artifacts/tmp/bin/p5_smem_probe"
STAGE_BIN="$ROOT/artifacts/tmp/bin/p5_smem_probe_stage"
STAGE_LOG="$ROOT/artifacts/tmp/p5_smem_probe_stage_compile.log"
CUTLASS_INCLUDE="$ROOT/.venv-trtllm/lib/python3.12/site-packages/flashinfer/data/cutlass/include"

mkdir -p "$(dirname "$BIN")"

nvcc \
  -std=c++20 \
  -arch=sm_120a \
  -I"$CUTLASS_INCLUDE" \
  -o "$BIN" \
  "$SRC"

"$BIN"

echo
echo "Compiling naive current+64KB staging variant..."
set +e
nvcc \
  -std=c++20 \
  -arch=sm_120a \
  -DNEMOTRON_INCLUDE_STAGE_KERNEL=1 \
  -I"$CUTLASS_INCLUDE" \
  -o "$STAGE_BIN" \
  "$SRC" \
  >"$STAGE_LOG" 2>&1
STAGE_STATUS=$?
set -e

if [[ "$STAGE_STATUS" -eq 0 ]]; then
  echo "naive current+64KB staging variant compiled successfully"
else
  echo "naive current+64KB staging variant failed to compile as expected"
  cat "$STAGE_LOG"
fi
