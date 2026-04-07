#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/home/khkramer/models/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1}
OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-${PLATFORM_ROOT}/artifacts/manifests/forward_runtime_manifest_unverified.json}
SOURCE_REVISION=${SOURCE_REVISION:-b1ffe4992d7db6d768453a551a656b8d12c638fb}
TOKENIZER_REVISION=${TOKENIZER_REVISION:-${SOURCE_REVISION}}
PACKER_VERSION=${PACKER_VERSION:-forward-manifest-v1-dev}

ARGS=(
  --model-dir "${MODEL_DIR}"
  --output-manifest "${OUTPUT_MANIFEST}"
  --source-revision "${SOURCE_REVISION}"
  --tokenizer-revision "${TOKENIZER_REVISION}"
  --packer-version "${PACKER_VERSION}"
)

if [[ "${COMPUTE_CHECKSUMS:-0}" == "1" ]]; then
  ARGS+=(--compute-checksums)
fi

python3 "${PLATFORM_ROOT}/tools/manifest/generate_forward_manifest.py" "${ARGS[@]}"
