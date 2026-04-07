#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO_ROOT="$(cd "${PLATFORM_ROOT}/../.." && pwd)"
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"${PLATFORM_ROOT}/artifacts/benchmarks"}
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

JSON_OUTPUT=${JSON_OUTPUT:-"${ARTIFACTS_DIR}/gb10_mamba_trace_family_${TIMESTAMP}.json"}
MARKDOWN_OUTPUT=${MARKDOWN_OUTPUT:-"${ARTIFACTS_DIR}/gb10_mamba_trace_family_${TIMESTAMP}.md"}

python3 "${PLATFORM_ROOT}/tools/benchmark_analysis/summarize_mamba_trace_family.py" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --json-output "${JSON_OUTPUT}" \
  --markdown-output "${MARKDOWN_OUTPUT}" \
  "$@"
