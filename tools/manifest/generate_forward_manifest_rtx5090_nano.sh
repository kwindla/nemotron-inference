#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-${ROOT_DIR}/artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}
OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-${ROOT_DIR}/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json}
MODEL_ID=${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}
PACKER_VERSION=${PACKER_VERSION:-forward-manifest-v1-dev}
GPU_FAMILY=${GPU_FAMILY:-RTX5090}
COMPUTE_CAPABILITY=${COMPUTE_CAPABILITY:-12.0}

ARGS=(
  --model-dir "${MODEL_DIR}"
  --output-manifest "${OUTPUT_MANIFEST}"
  --model-id "${MODEL_ID}"
  --packer-version "${PACKER_VERSION}"
  --gpu-family "${GPU_FAMILY}"
  --compute-capability "${COMPUTE_CAPABILITY}"
)

if [[ -n "${SOURCE_REVISION:-}" ]]; then
  ARGS+=(--source-revision "${SOURCE_REVISION}")
fi

if [[ -n "${TOKENIZER_REVISION:-}" ]]; then
  ARGS+=(--tokenizer-revision "${TOKENIZER_REVISION}")
fi

if [[ "${COMPUTE_CHECKSUMS:-0}" == "1" ]]; then
  ARGS+=(--compute-checksums)
fi

mkdir -p "$(dirname "${OUTPUT_MANIFEST}")"

python3 "${ROOT_DIR}/tools/manifest/generate_forward_manifest.py" "${ARGS[@]}"
