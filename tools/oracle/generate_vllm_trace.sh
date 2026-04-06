#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=${ROOT_DIR%/nemotron-runtime}
IMAGE=${IMAGE:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}
MODEL_DIR=${MODEL_DIR:-/models/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1}
OUTPUT_DIR=${OUTPUT_DIR:-proj-2026-04-05-0516/vllm-trace-prefill}
DECODE_STEPS=${DECODE_STEPS:-8}
DECODE_CAPTURE_LAYERS=${DECODE_CAPTURE_LAYERS:-0}
EXPERT_CAPTURE_LAYER=${EXPERT_CAPTURE_LAYER:-}
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
  python3 /workspace/nemotron-runtime/tools/oracle/dump_vllm_trace.py
  --model-dir "${MODEL_DIR}"
  --prompts-fixture "${PROMPTS_FIXTURE}"
  --decode-steps "${DECODE_STEPS}"
  --image-tag "${IMAGE}"
  --output-dir "${OUTPUT_DIR_IN_CONTAINER}"
)

if [[ "${DECODE_CAPTURE_LAYERS}" != "0" ]]; then
  container_cmd+=(--decode-capture-layers)
fi
if [[ -n "${EXPERT_CAPTURE_LAYER}" ]]; then
  container_cmd+=(--expert-capture-layer "${EXPERT_CAPTURE_LAYER}")
fi

printf -v container_cmd_str '%q ' "${container_cmd[@]}"

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v /home/khkramer/models:/models:ro \
  -v "${WORKSPACE_ROOT}:/workspace" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "${container_cmd_str}"
