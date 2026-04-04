#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_DIR=${MODEL_DIR:-${ROOT_DIR}/artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}
LAYER_INDEX=${LAYER_INDEX:-0}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/testing/oracle/mamba_layer${LAYER_INDEX}_decode_block}
INPUT_HIDDEN_BIN=${INPUT_HIDDEN_BIN:-}
INITIAL_CONV_STATE_BIN=${INITIAL_CONV_STATE_BIN:-}
INITIAL_SSM_STATE_BIN=${INITIAL_SSM_STATE_BIN:-}

INPUT_HIDDEN_ARG=""
INITIAL_CONV_STATE_ARG=""
INITIAL_SSM_STATE_ARG=""
if [[ -n "${INPUT_HIDDEN_BIN}" ]]; then
  INPUT_HIDDEN_ARG="--input-hidden-bin ${INPUT_HIDDEN_BIN}"
fi
if [[ -n "${INITIAL_CONV_STATE_BIN}" ]]; then
  INITIAL_CONV_STATE_ARG="--initial-conv-state-bin ${INITIAL_CONV_STATE_BIN}"
fi
if [[ -n "${INITIAL_SSM_STATE_BIN}" ]]; then
  INITIAL_SSM_STATE_ARG="--initial-ssm-state-bin ${INITIAL_SSM_STATE_BIN}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/oracle/dump_mamba_layer_fixture.py" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --layer-index "${LAYER_INDEX}" \
  ${INPUT_HIDDEN_ARG} \
  ${INITIAL_CONV_STATE_ARG} \
  ${INITIAL_SSM_STATE_ARG}
