#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO_ROOT=$(cd "${PLATFORM_ROOT}/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/models/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1}
LAYER_INDEX=${LAYER_INDEX:-0}
IMAGE=${IMAGE:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/nemotron-runtime/platforms/super-spark/testing/oracle/mamba_layer${LAYER_INDEX}_decode_block}
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

docker run --rm \
  -v /home/khkramer/models:/models:ro \
  -v "${REPO_ROOT%/nemotron-runtime}:/workspace" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "python3 /workspace/nemotron-runtime/platforms/super-spark/tools/oracle/dump_mamba_layer_fixture.py \
    --model-dir ${MODEL_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --layer-index ${LAYER_INDEX} \
    ${INPUT_HIDDEN_ARG} \
    ${INITIAL_CONV_STATE_ARG} \
    ${INITIAL_SSM_STATE_ARG}"
