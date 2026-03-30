#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"${ROOT_DIR}/artifacts/benchmarks"}
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

JSON_OUTPUT=${JSON_OUTPUT:-"${ARTIFACTS_DIR}/gb10_mamba_trace_compare_${TIMESTAMP}.json"}
MARKDOWN_OUTPUT=${MARKDOWN_OUTPUT:-"${ARTIFACTS_DIR}/gb10_mamba_trace_compare_${TIMESTAMP}.md"}

python3 "${ROOT_DIR}/tools/benchmark_analysis/compare_mamba_trace_benchmarks.py" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --baseline-label synthetic_phase_end \
  --candidate-label oracle_trace_v2 \
  --json-output "${JSON_OUTPUT}" \
  --markdown-output "${MARKDOWN_OUTPUT}" \
  "$@"
