#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/models/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1}
PROMPTS_PATH=${PROMPTS_PATH:-/workspace/nemotron-runtime/testing/oracle/prompts.json}
PROMPT_NAME=${PROMPT_NAME:-short_chat}
MODE=${MODE:-single_token}
TOKEN_INDEX=${TOKEN_INDEX:-0}
PROMPT_TOKEN_COUNT=${PROMPT_TOKEN_COUNT:-0}
CAPTURE_LAYERS=${CAPTURE_LAYERS:-0,1,7}
STOP_LAYER=${STOP_LAYER:--1}
IMAGE=${IMAGE:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/nemotron-runtime/testing/oracle/full_model_single_token_${PROMPT_NAME}}
ORACLE_DEVICE=${ORACLE_DEVICE:-cuda}

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v /home/khkramer/models:/models:ro \
  -v "${ROOT_DIR%/nemotron-runtime}:/workspace" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "python3 /workspace/nemotron-runtime/tools/oracle/dump_single_token_forward_fixture.py \
    --model-dir ${MODEL_DIR} \
    --prompts ${PROMPTS_PATH} \
    --prompt-name ${PROMPT_NAME} \
    --mode ${MODE} \
    --token-index ${TOKEN_INDEX} \
    --prompt-token-count ${PROMPT_TOKEN_COUNT} \
    --capture-layers ${CAPTURE_LAYERS} \
    --stop-layer ${STOP_LAYER} \
    --device ${ORACLE_DEVICE} \
    --output-dir ${OUTPUT_DIR}"
