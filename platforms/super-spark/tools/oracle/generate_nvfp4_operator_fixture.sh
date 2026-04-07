#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO_ROOT=$(cd "${PLATFORM_ROOT}/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/models/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1}
PROMPTS_PATH=${PROMPTS_PATH:-/workspace/nemotron-runtime/platforms/super-spark/testing/oracle/prompts.json}
PROMPT_NAME=${PROMPT_NAME:-short_chat}
OPERATOR_KIND=${OPERATOR_KIND:-up_proj}
ROWS=${ROWS:-16}
LAYER_INDEX=${LAYER_INDEX:-1}
EXPERT_INDEX=${EXPERT_INDEX:-0}
IMAGE=${IMAGE:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "${OPERATOR_KIND}" == shared_* ]]; then
    OUTPUT_DIR="/workspace/nemotron-runtime/platforms/super-spark/testing/oracle/nvfp4_layer${LAYER_INDEX}_${OPERATOR_KIND}"
  else
    OUTPUT_DIR="/workspace/nemotron-runtime/platforms/super-spark/testing/oracle/nvfp4_layer${LAYER_INDEX}_expert${EXPERT_INDEX}_${OPERATOR_KIND}"
  fi
fi

docker run --rm \
  -v /home/khkramer/models:/models:ro \
  -v "${REPO_ROOT%/nemotron-runtime}:/workspace" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "python3 /workspace/nemotron-runtime/platforms/super-spark/tools/oracle/dump_nvfp4_operator_fixture.py \
    --model-dir ${MODEL_DIR} \
    --prompts ${PROMPTS_PATH} \
    --prompt-name ${PROMPT_NAME} \
    --output-dir ${OUTPUT_DIR} \
    --rows ${ROWS} \
    --layer-index ${LAYER_INDEX} \
    --expert-index ${EXPERT_INDEX} \
    --operator-kind ${OPERATOR_KIND}"
