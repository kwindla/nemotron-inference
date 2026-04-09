#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-sm120-relwithdebinfo"
BENCH="${BUILD_DIR}/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench"
MANIFEST="${ROOT_DIR}/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${ROOT_DIR}/artifacts/profiles/routed_phase05_${TIMESTAMP}"
REPORT_BASE="${OUT_DIR}/ncu_cold_prefill_prefix128_unified_routed_fp4"
CSV_OUT="${OUT_DIR}/ncu_cold_prefill_prefix128_unified_routed_fp4.csv"
SUMMARY_OUT="${OUT_DIR}/ncu_cold_prefill_prefix128_unified_routed_fp4.summary.txt"
STDOUT_OUT="${OUT_DIR}/ncu_cold_prefill_prefix128_unified_routed_fp4.stdout.txt"
PERM_ERR="${OUT_DIR}/ncu_permission_check.stderr.txt"
NCU_STDERR="${OUT_DIR}/ncu_cold_prefill_prefix128_unified_routed_fp4.stderr.txt"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "run_phase05_routed_ncu.sh: manifest not found: ${MANIFEST}" >&2
  exit 1
fi

cmake --build "${BUILD_DIR}" --target nano_prefix_cache_ttft_bench -j 8

ncu --query-metrics >/dev/null 2>"${PERM_ERR}" || true
if rg -q "ERR_NVGPUCTRPERM" "${PERM_ERR}"; then
  echo "run_phase05_routed_ncu.sh: Nsight Compute is installed, but GPU performance counters are blocked on this machine." >&2
  echo "permission_check=${PERM_ERR}" >&2
  exit 2
fi
if [[ -s "${PERM_ERR}" ]]; then
  cat "${PERM_ERR}" >&2
  exit 1
fi

export NEMOTRON_FORWARD_MANIFEST="${MANIFEST}"

set +e
ncu --target-processes all \
  --kernel-name-base function \
  --kernel-name regex:Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel \
  --launch-count 1 \
  --set full \
  -o "${REPORT_BASE}" \
  "${BENCH}" \
    --moe-prefill-window-tokens 4096 \
    --warmup 0 \
    --iterations 5 \
    --tail-token-count 4 \
    --prefix-length 128 \
    --case cold_prefill_prefix128 \
    2> >(tee "${NCU_STDERR}" >&2) | tee "${STDOUT_OUT}"
ncu_status=${PIPESTATUS[0]}
set -e

if (( ncu_status != 0 )); then
  if rg -q "ERR_NVGPUCTRPERM" "${NCU_STDERR}" "${STDOUT_OUT}"; then
    echo "run_phase05_routed_ncu.sh: profiling failed because GPU performance counters are blocked on this machine." >&2
    echo "ncu_stderr=${NCU_STDERR}" >&2
    echo "stdout=${STDOUT_OUT}" >&2
    exit 2
  fi
  exit "${ncu_status}"
fi

ncu --import "${REPORT_BASE}.ncu-rep" --csv --page raw > "${CSV_OUT}"

python3 "${ROOT_DIR}/proj-2026-04-05-1704/summarize_phase05_ncu_csv.py" \
  "${CSV_OUT}" | tee "${SUMMARY_OUT}"

echo "report=${REPORT_BASE}.ncu-rep"
echo "csv=${CSV_OUT}"
echo "summary=${SUMMARY_OUT}"
echo "stdout=${STDOUT_OUT}"
