#!/usr/bin/env bash
set -euo pipefail

PLATFORM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${PLATFORM_ROOT}/../.." && pwd)"
ARTIFACT_DIR="${PLATFORM_ROOT}/artifacts/benchmarks"
GENERATED_DIR="${ARTIFACT_DIR}/generated_fixtures"
SOURCE_FIXTURE_ROOT="${SOURCE_FIXTURE_ROOT:-${PLATFORM_ROOT}/testing/oracle/mamba_layer0_target_chat_trace_markdown_headerless}"
PHASE_NAME="${PHASE_NAME:-user_turn_1_tail_prefill}"
DT_SCALES="${DT_SCALES:-0.75,1.0,1.25,1.5}"
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
SUMMARY_PREFIX="${ARTIFACT_DIR}/gb10_mamba_trace_dt_scale_sweep_${TIMESTAMP}_${CUDA_TAG:-cuda_unknown}"
SUMMARY_JSON="${SUMMARY_PREFIX}.json"
SUMMARY_MD="${SUMMARY_PREFIX}.md"
ENV_OUT="${SUMMARY_PREFIX}.env.txt"

mkdir -p "${ARTIFACT_DIR}" "${GENERATED_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=${PLATFORM_ROOT}"
  echo "source_fixture_root=${SOURCE_FIXTURE_ROOT}"
  echo "phase_name=${PHASE_NAME}"
  echo "dt_scales=${DT_SCALES}"
  echo "concurrency=${CONCURRENCY}"
  echo "binary=${REPO_ROOT}/build/platforms/super-spark/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench"
  echo "uname=$(uname -a)"
  echo
  echo "[nvcc]"
  nvcc --version || true
} > "${ENV_OUT}"

phase_length="$(
python3 - <<'PY' "${SOURCE_FIXTURE_ROOT}" "${PHASE_NAME}"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
phase_name = sys.argv[2]
metadata = json.loads((root / "metadata.json").read_text())
phase = next(item for item in metadata["trace_phases"] if item["name"] == phase_name)
print(int(phase["length"]))
PY
)"

INPUT_ARGS=()
IFS=',' read -r -a SCALE_ARRAY <<< "${DT_SCALES}"
for scale in "${SCALE_ARRAY[@]}"; do
  scale_tag="$(python3 - <<'PY' "${scale}"
import sys
value = float(sys.argv[1])
print(f"{value:.2f}".replace('.', 'p'))
PY
)"
  fixture_root="${GENERATED_DIR}/mamba_phase_${PHASE_NAME}_dt_${scale_tag}"
  python3 "${PLATFORM_ROOT}/tools/oracle/make_mamba_trace_phase_dt_variant.py" \
    --source-root "${SOURCE_FIXTURE_ROOT}" \
    --output-root "${fixture_root}" \
    --phase-name "${PHASE_NAME}" \
    --dt-scale "${scale}"

  artifact_prefix="${ARTIFACT_DIR}/gb10_mamba_trace_dt_scale_${scale_tag}_${TIMESTAMP}_${CUDA_TAG:-cuda_unknown}"
  json_out="${artifact_prefix}.json"
  stdout_out="${artifact_prefix}.stdout.txt"

  "${REPO_ROOT}/build/platforms/super-spark/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench" \
    --operations fixture_trace \
    --concurrency "${CONCURRENCY}" \
    --trace-fixture-root "${fixture_root}" \
    --fixture-trace-steps "${phase_length}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --json-output "${json_out}" | tee "${stdout_out}"

  INPUT_ARGS+=("--input" "${scale}=${json_out}|${fixture_root}")
done

python3 "${PLATFORM_ROOT}/tools/benchmark_analysis/analyze_mamba_trace_dt_scale_sweep.py" \
  "${INPUT_ARGS[@]}" \
  --phase-name "${PHASE_NAME}" \
  --json-output "${SUMMARY_JSON}" \
  --markdown-output "${SUMMARY_MD}"

echo "summary_json=${SUMMARY_JSON}"
echo "summary_md=${SUMMARY_MD}"
