#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO_ROOT=$(cd "${PLATFORM_ROOT}/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/models/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1}
LAYER_INDEX=${LAYER_INDEX:-0}
IMAGE=${IMAGE:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/nemotron-runtime/platforms/super-spark/testing/oracle/mamba_layer${LAYER_INDEX}_target_chat_trace}
PROFILE=${PROFILE:-/workspace/nemotron-runtime/platforms/super-spark/testing/oracle/mamba_target_chat_profile.json}

docker run --rm \
  -v /home/khkramer/models:/models:ro \
  -v "${REPO_ROOT%/nemotron-runtime}:/workspace" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "python3 /workspace/nemotron-runtime/platforms/super-spark/tools/oracle/dump_mamba_target_trace_fixture.py \
    --model-dir ${MODEL_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --profile ${PROFILE} \
    --layer-index ${LAYER_INDEX}"
