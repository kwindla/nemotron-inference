#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/benchmarks"
MANIFEST_PATH="${NEMOTRON_FORWARD_MANIFEST:-${ROOT_DIR}/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json}"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

BUILD_DIR="${NEMOTRON_BUILD_DIR:-}"
if [[ -z "${BUILD_DIR}" ]]; then
  for candidate in build-sm120-relwithdebinfo; do
    if [[ -x "${ROOT_DIR}/${candidate}/benchmarks/nano_fused_decode/nano_fused_decode_bench" ]]; then
      BUILD_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${BUILD_DIR}" ]]; then
  echo "run_default_bench.sh: nano_fused_decode_bench not found; set NEMOTRON_BUILD_DIR or build the benchmark first" >&2
  exit 1
fi

BINARY="${ROOT_DIR}/${BUILD_DIR}/benchmarks/nano_fused_decode/nano_fused_decode_bench"
BENCH_ARGS=("$@")
MODE_TAG="phased"
USER_JSON_OUT=""
for ((i = 0; i < ${#BENCH_ARGS[@]}; ++i)); do
  case "${BENCH_ARGS[$i]}" in
    --mode=*)
      MODE_TAG="${BENCH_ARGS[$i]#--mode=}"
      ;;
    --mode)
      if ((i + 1 < ${#BENCH_ARGS[@]})); then
        MODE_TAG="${BENCH_ARGS[$((i + 1))]}"
      fi
      ;;
    --json-output=*)
      USER_JSON_OUT="${BENCH_ARGS[$i]#--json-output=}"
      ;;
    --json-output)
      if ((i + 1 < ${#BENCH_ARGS[@]})); then
        USER_JSON_OUT="${BENCH_ARGS[$((i + 1))]}"
      fi
      ;;
  esac
done
MODE_TAG="${MODE_TAG//-/_}"
CUDA_TAG="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/cuda\1\2/p' | head -n1)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PREFIX="${ARTIFACT_DIR}/nano_fused_decode_16tok_${MODE_TAG}_${TIMESTAMP}_${CUDA_TAG:-cuda_unknown}"
JSON_OUT="${USER_JSON_OUT:-${OUTPUT_PREFIX}.json}"
STDOUT_OUT="${OUTPUT_PREFIX}.stdout.txt"
ENV_OUT="${OUTPUT_PREFIX}.env.txt"

mkdir -p "${ARTIFACT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cwd=${ROOT_DIR}"
  echo "build_dir=${BUILD_DIR}"
  echo "binary=${BINARY}"
  echo "manifest=${MANIFEST_PATH}"
  echo "uname=$(uname -a)"
  echo
  echo "[nvcc]"
  nvcc --version || true
  echo
  echo "[nvidia-smi]"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi
  echo
  echo "[cuda-install-dirs]"
  ls -d /usr/local/cuda* 2>/dev/null || true
} > "${ENV_OUT}"

BENCH_CMD=("${BINARY}" "--manifest" "${MANIFEST_PATH}")
if [[ -z "${USER_JSON_OUT}" ]]; then
  BENCH_CMD+=("--json-output" "${JSON_OUT}")
fi
BENCH_CMD+=("${BENCH_ARGS[@]}")

"${BENCH_CMD[@]}" | tee "${STDOUT_OUT}"

python3 - "${JSON_OUT}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

benchmark = data["benchmark"]
linear = data["linear_op_counters"]
expert = data["expert_staging_counters"]

required_linear = (
    "dense_fastpath_plan_success",
    "dense_fastpath_plan_fail",
    "dense_fastpath_execute",
    "dense_fastpath_execute_fail",
    "nvfp4_fastpath_plan_success",
    "nvfp4_fastpath_plan_fail",
    "nvfp4_fastpath_execute",
    "nvfp4_fastpath_execute_fail",
    "scaled_fp8_fastpath_execute",
)
required_expert = (
    "total_bytes_uploaded",
    "total_experts_staged",
    "total_staging_calls",
    "staging_elapsed_us",
)

for field in ("mode",):
    if field not in benchmark:
        raise KeyError(f"benchmark.{field}")
for field in required_linear:
    if field not in linear:
        raise KeyError(f"linear_op_counters.{field}")
for field in required_expert:
    if field not in expert:
        raise KeyError(f"expert_staging_counters.{field}")
if not isinstance(data["cudnn_fe_available"], bool):
    raise TypeError("cudnn_fe_available must be a boolean")

print(f"run_default_bench.sh: verified JSON artifact {path}")
PY
