#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/benchmarks"
ORACLE_DIR="${ROOT_DIR}/testing/oracle"
FIXTURE_REGEX="${FIXTURE_REGEX:-^mamba_layer0_target_chat_trace($|_)}"
THRESHOLDS="${THRESHOLDS:-inf,0.5,0.6,0.65,0.7,0.75,0.8}"
CONCURRENCY="${CONCURRENCY:-1,8}"
WARMUP="${WARMUP:-1}"
ITERATIONS="${ITERATIONS:-3}"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

CUDA_TAG="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/cuda\1\2/p' | head -n1)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PREFIX="${ARTIFACT_DIR}/gb10_mamba_dt_threshold_sweep_${TIMESTAMP}_${CUDA_TAG:-cuda_unknown}"
SUMMARY_JSON="${PREFIX}.json"
SUMMARY_MD="${PREFIX}.md"
ENV_OUT="${PREFIX}.env.txt"

mkdir -p "${ARTIFACT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=${ROOT_DIR}"
  echo "fixture_regex=${FIXTURE_REGEX}"
  echo "thresholds=${THRESHOLDS}"
  echo "concurrency=${CONCURRENCY}"
  echo "binary=${ROOT_DIR}/build/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench"
  echo "uname=$(uname -a)"
  echo
  echo "[nvcc]"
  nvcc --version || true
} > "${ENV_OUT}"

mapfile -t FIXTURES < <(ls -1 "${ORACLE_DIR}" | rg "${FIXTURE_REGEX}" | sort)
if [[ ${#FIXTURES[@]} -eq 0 ]]; then
  echo "no fixtures matched ${FIXTURE_REGEX}" >&2
  exit 1
fi

ANALYSIS_ARGS=()
IFS=',' read -r -a THRESHOLD_VALUES <<< "${THRESHOLDS}"
for threshold in "${THRESHOLD_VALUES[@]}"; do
  for fixture in "${FIXTURES[@]}"; do
    fixture_root="${ORACLE_DIR}/${fixture}"
    threshold_tag="${threshold//./p}"
    json_out="${ARTIFACT_DIR}/gb10_mamba_dt_threshold_${threshold_tag}_${fixture}_${TIMESTAMP}_${CUDA_TAG:-cuda_unknown}.json"
    stdout_out="${ARTIFACT_DIR}/gb10_mamba_dt_threshold_${threshold_tag}_${fixture}_${TIMESTAMP}_${CUDA_TAG:-cuda_unknown}.stdout.txt"
    fixture_steps="$(
      python3 - <<'PY' "${fixture_root}"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text())
print(",".join(str(int(item["end_step"])) for item in metadata["trace_phases"]))
PY
    )"

    cmd=(
      "${ROOT_DIR}/build/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench"
      --operations fixture_trace
      --concurrency "${CONCURRENCY}"
      --trace-fixture-root "${fixture_root}"
      --fixture-trace-steps "${fixture_steps}"
      --warmup "${WARMUP}"
      --iterations "${ITERATIONS}"
      --json-output "${json_out}"
    )
    if [[ "${threshold}" != "inf" ]]; then
      cmd+=(--dt-threshold "${threshold}")
    fi

    "${cmd[@]}" > "${stdout_out}"
    ANALYSIS_ARGS+=(--input "${threshold}=${json_out}")
  done
done

python3 "${ROOT_DIR}/tools/benchmark_analysis/analyze_mamba_dt_threshold_sweep.py" \
  "${ANALYSIS_ARGS[@]}" \
  --json-output "${SUMMARY_JSON}" \
  --markdown-output "${SUMMARY_MD}"

echo "summary_json=${SUMMARY_JSON}"
echo "summary_md=${SUMMARY_MD}"
