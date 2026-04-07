#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO_ROOT="$(cd "${PLATFORM_ROOT}/../.." && pwd)"
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"${PLATFORM_ROOT}/artifacts/benchmarks"}
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

JSON_OUTPUT=${JSON_OUTPUT:-"${ARTIFACTS_DIR}/gb10_mamba_trace_compare_${TIMESTAMP}.json"}
MARKDOWN_OUTPUT=${MARKDOWN_OUTPUT:-"${ARTIFACTS_DIR}/gb10_mamba_trace_compare_${TIMESTAMP}.md"}

python3 "${PLATFORM_ROOT}/tools/benchmark_analysis/compare_mamba_trace_benchmarks.py" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --baseline-label synthetic_phase_end \
  --candidate-label oracle_trace_v2 \
  --json-output "${JSON_OUTPUT}" \
  --markdown-output "${MARKDOWN_OUTPUT}" \
  "$@"
