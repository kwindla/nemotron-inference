#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_JSON="${SCRIPT_DIR}/vllm_baseline_results.json"
PYTHON_BIN="${REPO_ROOT}/vllm-env/bin/python"

"${PYTHON_BIN}" "${SCRIPT_DIR}/bench_vllm_nano.py" --output "${OUTPUT_JSON}" "$@"
