#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO_ROOT=$(cd "${PLATFORM_ROOT}/../.." && pwd)
WORKSPACE_ROOT=${REPO_ROOT%/nemotron-runtime}
IMAGE=${IMAGE:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}
INPUT_DIR=${INPUT_DIR:-proj-2026-04-05-vllm-parity-reset/chunked-scan-vllm-160}
OUTPUT_DIR=${OUTPUT_DIR:-proj-2026-04-05-vllm-parity-reset/chunk-cumsum-kernel-artifacts}
CHUNK_SIZE=${CHUNK_SIZE:-128}
DT_DTYPE=${DT_DTYPE:-fp32}

resolve_workspace_path() {
  local path=$1
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
    return
  fi
  path=${path#./}
  printf '/workspace/%s\n' "${path}"
}

INPUT_DIR_IN_CONTAINER=$(resolve_workspace_path "${INPUT_DIR}")
OUTPUT_DIR_IN_CONTAINER=$(resolve_workspace_path "${OUTPUT_DIR}")

container_cmd=(
  python3 /workspace/nemotron-runtime/platforms/super-spark/tools/oracle/dump_vllm_chunk_cumsum_kernel_artifacts.py
  --input-dir "${INPUT_DIR_IN_CONTAINER}"
  --output-dir "${OUTPUT_DIR_IN_CONTAINER}"
  --chunk-size "${CHUNK_SIZE}"
  --dt-dtype "${DT_DTYPE}"
)

printf -v container_cmd_str '%q ' "${container_cmd[@]}"

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${WORKSPACE_ROOT}:/workspace" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "${container_cmd_str}"
