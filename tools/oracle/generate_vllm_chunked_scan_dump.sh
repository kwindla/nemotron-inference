#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=${ROOT_DIR%/nemotron-runtime}
IMAGE=${IMAGE:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}
MODEL_DIR=${MODEL_DIR:-/models/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1}
OUTPUT_DIR=${OUTPUT_DIR:-proj-2026-04-05-1814/dumps/vllm}
TARGET_LAYER=${TARGET_LAYER:-0}
PROMPTS_FIXTURE=/workspace/nemotron-runtime/testing/oracle/full_model_single_token_short_chat_cuda_v3/prompt_token_ids.json

resolve_workspace_path() {
  local path=$1
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
    return
  fi
  path=${path#./}
  printf '/workspace/%s\n' "${path}"
}

OUTPUT_DIR_IN_CONTAINER=$(resolve_workspace_path "${OUTPUT_DIR}")

container_cmd=(
  python3 /workspace/nemotron-runtime/tools/oracle/dump_vllm_chunked_scan_intermediates.py
  --model-dir "${MODEL_DIR}"
  --prompts-fixture "${PROMPTS_FIXTURE}"
  --target-layer "${TARGET_LAYER}"
  --image-tag "${IMAGE}"
  --output-dir "${OUTPUT_DIR_IN_CONTAINER}"
)

printf -v container_cmd_str '%q ' "${container_cmd[@]}"

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v /home/khkramer/models:/models:ro \
  -v "${WORKSPACE_ROOT}:/workspace" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "${container_cmd_str}"
