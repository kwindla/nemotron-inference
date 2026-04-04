#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_DIR=${MODEL_DIR:-${ROOT_DIR}/artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}
LAYER_INDEX=${LAYER_INDEX:-0}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/testing/oracle/mamba_layer${LAYER_INDEX}_decode_update}

"${PYTHON_BIN}" "${ROOT_DIR}/tools/oracle/dump_mamba_update_fixture.py" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --layer-index "${LAYER_INDEX}"
