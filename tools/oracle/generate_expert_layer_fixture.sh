#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/models/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1}
LAYER_INDEX=${LAYER_INDEX:-1}
IMAGE=${IMAGE:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/nemotron-runtime/testing/oracle/expert_layer${LAYER_INDEX}_decode_block}
INPUT_HIDDEN_BIN=${INPUT_HIDDEN_BIN:-}
ORACLE_DEVICE=${ORACLE_DEVICE:-cuda}

INPUT_HIDDEN_ARG=""
if [[ -n "${INPUT_HIDDEN_BIN}" ]]; then
  INPUT_HIDDEN_ARG="--input-hidden-bin ${INPUT_HIDDEN_BIN}"
fi

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v /home/khkramer/models:/models:ro \
  -v "${ROOT_DIR%/nemotron-runtime}:/workspace" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "python3 /workspace/nemotron-runtime/tools/oracle/dump_expert_layer_fixture.py \
    --model-dir ${MODEL_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --layer-index ${LAYER_INDEX} \
    --device ${ORACLE_DEVICE} \
    ${INPUT_HIDDEN_ARG}"
