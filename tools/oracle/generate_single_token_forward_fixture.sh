#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_DIR=${MODEL_DIR:-${ROOT_DIR}/artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}
PROMPTS_PATH=${PROMPTS_PATH:-${ROOT_DIR}/testing/oracle/prompts.json}
PROMPT_NAME=${PROMPT_NAME:-short_chat}
MODE=${MODE:-single_token}
TOKEN_INDEX=${TOKEN_INDEX:-0}
PROMPT_TOKEN_COUNT=${PROMPT_TOKEN_COUNT:-0}
CAPTURE_LAYERS=${CAPTURE_LAYERS:-0,1,7}
STOP_LAYER=${STOP_LAYER:--1}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/testing/oracle/full_model_single_token_${PROMPT_NAME}_cuda_v3}
ORACLE_DEVICE=${ORACLE_DEVICE:-cuda}

"${PYTHON_BIN}" "${ROOT_DIR}/tools/oracle/dump_single_token_forward_fixture.py" \
  --model-dir "${MODEL_DIR}" \
  --prompts "${PROMPTS_PATH}" \
  --prompt-name "${PROMPT_NAME}" \
  --mode "${MODE}" \
  --token-index "${TOKEN_INDEX}" \
  --prompt-token-count "${PROMPT_TOKEN_COUNT}" \
  --capture-layers "${CAPTURE_LAYERS}" \
  --stop-layer "${STOP_LAYER}" \
  --device "${ORACLE_DEVICE}" \
  --output-dir "${OUTPUT_DIR}"
