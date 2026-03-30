#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"${ROOT_DIR}/artifacts/benchmarks"}
ACTIVATION_STAGING=${ACTIVATION_STAGING:-device}
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

JSON_OUTPUT=${JSON_OUTPUT:-"${ARTIFACTS_DIR}/gb10_gemm_compare_${TIMESTAMP}_${ACTIVATION_STAGING}.json"}
MARKDOWN_OUTPUT=${MARKDOWN_OUTPUT:-"${ARTIFACTS_DIR}/gb10_gemm_compare_${TIMESTAMP}_${ACTIVATION_STAGING}.md"}

python3 "${ROOT_DIR}/tools/benchmark_analysis/compare_gemm_benchmarks.py" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --activation-staging "${ACTIVATION_STAGING}" \
  --json-output "${JSON_OUTPUT}" \
  --markdown-output "${MARKDOWN_OUTPUT}" \
  "$@"
