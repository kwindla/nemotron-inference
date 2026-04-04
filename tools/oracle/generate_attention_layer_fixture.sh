#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_DIR=${MODEL_DIR:-${ROOT_DIR}/artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}
PROMPTS_PATH=${PROMPTS_PATH:-${ROOT_DIR}/testing/oracle/prompts.json}
PROMPT_NAME=${PROMPT_NAME:-short_chat}
LAYER_INDEX=${LAYER_INDEX:-7}
TOKENS_PER_PAGE=${TOKENS_PER_PAGE:-16}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/testing/oracle/attention_layer${LAYER_INDEX}_${PROMPT_NAME}}
INPUT_HIDDEN_BIN=${INPUT_HIDDEN_BIN:-}

INPUT_HIDDEN_ARG=""
if [[ -n "${INPUT_HIDDEN_BIN}" ]]; then
  INPUT_HIDDEN_ARG="--input-hidden-bin ${INPUT_HIDDEN_BIN}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/oracle/dump_attention_layer_fixture.py" \
  --model-dir "${MODEL_DIR}" \
  --prompts "${PROMPTS_PATH}" \
  --prompt-name "${PROMPT_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --layer-index "${LAYER_INDEX}" \
  --tokens-per-page "${TOKENS_PER_PAGE}" \
  ${INPUT_HIDDEN_ARG}
