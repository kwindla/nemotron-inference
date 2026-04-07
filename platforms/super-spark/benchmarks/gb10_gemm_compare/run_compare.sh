#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO_ROOT="$(cd "${PLATFORM_ROOT}/../.." && pwd)"
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"${PLATFORM_ROOT}/artifacts/benchmarks"}
ACTIVATION_STAGING=${ACTIVATION_STAGING:-device}
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

JSON_OUTPUT=${JSON_OUTPUT:-"${ARTIFACTS_DIR}/gb10_gemm_compare_${TIMESTAMP}_${ACTIVATION_STAGING}.json"}
MARKDOWN_OUTPUT=${MARKDOWN_OUTPUT:-"${ARTIFACTS_DIR}/gb10_gemm_compare_${TIMESTAMP}_${ACTIVATION_STAGING}.md"}

python3 "${PLATFORM_ROOT}/tools/benchmark_analysis/compare_gemm_benchmarks.py" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --activation-staging "${ACTIVATION_STAGING}" \
  --json-output "${JSON_OUTPUT}" \
  --markdown-output "${MARKDOWN_OUTPUT}" \
  "$@"
