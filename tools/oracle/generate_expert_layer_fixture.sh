#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_DIR=${MODEL_DIR:-${ROOT_DIR}/artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}
LAYER_INDEX=${LAYER_INDEX:-1}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/testing/oracle/expert_layer${LAYER_INDEX}_decode_block}
INPUT_HIDDEN_BIN=${INPUT_HIDDEN_BIN:-}
ORACLE_DEVICE=${ORACLE_DEVICE:-cuda}

INPUT_HIDDEN_ARG=""
if [[ -n "${INPUT_HIDDEN_BIN}" ]]; then
  INPUT_HIDDEN_ARG="--input-hidden-bin ${INPUT_HIDDEN_BIN}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/oracle/dump_expert_layer_fixture.py" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --layer-index "${LAYER_INDEX}" \
  --device "${ORACLE_DEVICE}" \
  ${INPUT_HIDDEN_ARG}
