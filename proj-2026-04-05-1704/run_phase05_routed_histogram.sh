#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-sm120-relwithdebinfo"
BENCH="${BUILD_DIR}/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench"
MANIFEST="${ROOT_DIR}/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAW_OUT="${ROOT_DIR}/artifacts/benchmarks/routed_phase05_prefix128_${TIMESTAMP}.jsonl"
SUMMARY_OUT="${ROOT_DIR}/artifacts/benchmarks/routed_phase05_prefix128_${TIMESTAMP}.summary.txt"
STDOUT_OUT="${ROOT_DIR}/artifacts/benchmarks/routed_phase05_prefix128_${TIMESTAMP}.stdout.txt"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "run_phase05_routed_histogram.sh: manifest not found: ${MANIFEST}" >&2
  exit 1
fi

cmake --build "${BUILD_DIR}" --target nano_prefix_cache_ttft_bench -j 8

export NEMOTRON_FORWARD_MANIFEST="${MANIFEST}"
export NEMOTRON_ROUTED_PHASE05_DUMP="${RAW_OUT}"

"${BENCH}" \
  --moe-prefill-window-tokens 4096 \
  --warmup 0 \
  --iterations 5 \
  --tail-token-count 4 \
  --prefix-length 128 \
  --prefix-length 4096 \
  --case cold_prefill_prefix128 | tee "${STDOUT_OUT}"

python3 "${ROOT_DIR}/proj-2026-04-05-1704/summarize_routed_phase05_histogram.py" \
  "${RAW_OUT}" | tee "${SUMMARY_OUT}"

echo "raw_histogram=${RAW_OUT}"
echo "summary=${SUMMARY_OUT}"
echo "stdout=${STDOUT_OUT}"
