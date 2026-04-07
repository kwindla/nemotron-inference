#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_REF="${VLLM_REF:-b31e9326a}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.5}"
CUDA_VERSION="${CUDA_VERSION:-13.0.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
FINAL_BASE_IMAGE="${FINAL_BASE_IMAGE:-${BUILD_BASE_IMAGE}}"
MAX_JOBS="${MAX_JOBS:-8}"
NVCC_THREADS="${NVCC_THREADS:-2}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"
DOCKER_TARGET="${DOCKER_TARGET:-vllm-openai}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${REPO_ROOT}/Dockerfile.local-vllm}"
LOCAL_VLLM_VERSION="${LOCAL_VLLM_VERSION:-0.17.1.dev0+g${VLLM_REF}.fi065}"
LOCAL_IMAGE_TAG="${LOCAL_IMAGE_TAG:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}"
UPSTREAM_REPO_DIR="${UPSTREAM_REPO_DIR:-${REPO_ROOT}/.cache/vllm}"
WORKTREE_DIR="${WORKTREE_DIR:-${REPO_ROOT}/.cache/vllm-build-${VLLM_REF}}"
DRY_RUN=0

usage() {
  cat <<EOF
Usage: ./build_local_vllm_image.sh [--dry-run]

Builds a local arm64 vLLM image from upstream source using the repo-owned
Dockerfile pinned in this repo.

Default output image:
  ${LOCAL_IMAGE_TAG}

Important defaults:
  VLLM_REF=${VLLM_REF}
  FLASHINFER_VERSION=${FLASHINFER_VERSION}
  CUDA_VERSION=${CUDA_VERSION}
  BUILD_BASE_IMAGE=${BUILD_BASE_IMAGE}
  FINAL_BASE_IMAGE=${FINAL_BASE_IMAGE}
  MAX_JOBS=${MAX_JOBS}
  NVCC_THREADS=${NVCC_THREADS}
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
  DOCKERFILE_PATH=${DOCKERFILE_PATH}

Examples:
  ./build_local_vllm_image.sh
  LOCAL_IMAGE_TAG=nemotron-local/dgx-spark-vllm:dev ./build_local_vllm_image.sh
  MAX_JOBS=4 NVCC_THREADS=1 ./build_local_vllm_image.sh
EOF
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

prepare_upstream_repo() {
  mkdir -p "$(dirname "${UPSTREAM_REPO_DIR}")"
  if [ ! -d "${UPSTREAM_REPO_DIR}/.git" ]; then
    git clone https://github.com/vllm-project/vllm.git "${UPSTREAM_REPO_DIR}"
  fi

  git -C "${UPSTREAM_REPO_DIR}" fetch --all --tags
  git -C "${UPSTREAM_REPO_DIR}" worktree prune

  if [ -e "${WORKTREE_DIR}" ]; then
    git -C "${UPSTREAM_REPO_DIR}" worktree remove --force "${WORKTREE_DIR}" 2>/dev/null || rm -rf "${WORKTREE_DIR}"
  fi

  mkdir -p "$(dirname "${WORKTREE_DIR}")"
  git -C "${UPSTREAM_REPO_DIR}" worktree add --force --detach "${WORKTREE_DIR}" "${VLLM_REF}"
}

patch_flashinfer_version() {
  local cuda_requirements="${WORKTREE_DIR}/requirements/cuda.txt"
  python3 - "${cuda_requirements}" "${FLASHINFER_VERSION}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
flashinfer_version = sys.argv[2]
text = path.read_text()
updated = re.sub(
    r"^flashinfer-python==[^\n]+$",
    f"flashinfer-python=={flashinfer_version}",
    text,
    flags=re.MULTILINE,
)
if text == updated:
    raise SystemExit(f"Did not find flashinfer-python pin in {path}")
path.write_text(updated)
PY
}

prepare_local_dockerfile() {
  cp "${DOCKERFILE_PATH}" "${WORKTREE_DIR}/Dockerfile.local-vllm"
}

print_config() {
  cat <<EOF
VLLM_REF=${VLLM_REF}
FLASHINFER_VERSION=${FLASHINFER_VERSION}
CUDA_VERSION=${CUDA_VERSION}
PYTHON_VERSION=${PYTHON_VERSION}
BUILD_BASE_IMAGE=${BUILD_BASE_IMAGE}
FINAL_BASE_IMAGE=${FINAL_BASE_IMAGE}
MAX_JOBS=${MAX_JOBS}
NVCC_THREADS=${NVCC_THREADS}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
DOCKER_TARGET=${DOCKER_TARGET}
DOCKERFILE_PATH=${DOCKERFILE_PATH}
LOCAL_VLLM_VERSION=${LOCAL_VLLM_VERSION}
LOCAL_IMAGE_TAG=${LOCAL_IMAGE_TAG}
UPSTREAM_REPO_DIR=${UPSTREAM_REPO_DIR}
WORKTREE_DIR=${WORKTREE_DIR}
EOF
}

build_image() {
  DOCKER_BUILDKIT=1 docker build "${WORKTREE_DIR}" \
    --file "${WORKTREE_DIR}/Dockerfile.local-vllm" \
    --target "${DOCKER_TARGET}" \
    --platform "linux/arm64" \
    --tag "${LOCAL_IMAGE_TAG}" \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg FLASHINFER_VERSION="${FLASHINFER_VERSION}" \
    --build-arg BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE}" \
    --build-arg FINAL_BASE_IMAGE="${FINAL_BASE_IMAGE}" \
    --build-arg max_jobs="${MAX_JOBS}" \
    --build-arg nvcc_threads="${NVCC_THREADS}" \
    --build-arg torch_cuda_arch_list="${TORCH_CUDA_ARCH_LIST}" \
    --build-arg VLLM_VERSION_OVERRIDE="${LOCAL_VLLM_VERSION}" \
    --build-arg RUN_WHEEL_CHECK=false
}

prepare_upstream_repo
patch_flashinfer_version
prepare_local_dockerfile
print_config

if [ "${DRY_RUN}" -eq 1 ]; then
  exit 0
fi

build_image

cat <<EOF
Built image: ${LOCAL_IMAGE_TAG}

Suggested launch:
  MAX_JOBS=2 \\
  SUPER_VLLM_USE_FLASHINFER_MOE_FP4=0 \\
  SUPER_VLLM_USE_FLASHINFER_MOE_FP8=0 \\
  ./run_nemotron_nvfp4.sh super --image ${LOCAL_IMAGE_TAG}
EOF
