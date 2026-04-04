#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bench_full_comparison.sh [options]

Run the step-7 comparison harness across:
  - current runtime baseline
  - unified fused runtime
  - vLLM flashinfer_cutlass baseline
  - explicit unified-disable fallback checks

Outputs one timestamped artifact directory containing raw logs, JSON artifacts, and
summary tables.

Options:
  --manifest PATH            Manifest path. Defaults to NEMOTRON_FORWARD_MANIFEST or
                             artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json
  --bench-build-dir DIR      Build directory containing nano_fused_decode_bench and
                             nano_prefix_cache_ttft_bench. Defaults to auto-detect.
  --test-build-dir DIR       Build directory containing nano_save_prompt_oracle.
                             Defaults to auto-detect.
  --artifact-dir DIR         Output directory root. Default:
                             proj-2026-04-03-0318/artifacts/full_comparison
  --decode-tokens N          Steady-state decode steps for nano_fused_decode_bench.
                             Default: 16
  --prefix-warmup N          Warmup iterations for nano_prefix_cache_ttft_bench.
                             Default: 1
  --prefix-iters N           Measured iterations for nano_prefix_cache_ttft_bench.
                             Default: 5
  --vllm-warmup-iters N      Warmup iterations for bench_vllm_nano.py. Default: 2
  --vllm-iters N             Measured iterations for bench_vllm_nano.py. Default: 5
  --vllm-decode-prefix N     Prefix-cache seed prompt length for the vLLM decode case.
                             Default: 256
  --label TEXT               Optional suffix for the artifact directory name.
  --help, -h                 Show this message.

Notes:
  - The prefix-cache TTFT matrix records real 32/64/128-token resumed-tail cases by
    driving nano_prefix_cache_ttft_bench with matching
    --tail-token-count/--moe-prefill-window-tokens values.
  - The fallback check uses nano_save_prompt_oracle to verify that
    NEMOTRON_FORWARD_UNIFIED_FUSED=0 overrides the legacy opt-in and preserves the
    baseline output sequence.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MANIFEST="${REPO_ROOT}/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json"
MANIFEST_PATH="${NEMOTRON_FORWARD_MANIFEST:-${DEFAULT_MANIFEST}}"
BENCH_BUILD_DIR="${NEMOTRON_BENCH_BUILD_DIR:-}"
TEST_BUILD_DIR="${NEMOTRON_TEST_BUILD_DIR:-}"
ARTIFACT_ROOT="${SCRIPT_DIR}/artifacts/full_comparison"
DECODE_TOKENS="${DECODE_TOKENS:-16}"
PREFIX_WARMUP="${PREFIX_WARMUP:-1}"
PREFIX_ITERS="${PREFIX_ITERS:-5}"
VLLM_WARMUP_ITERS="${VLLM_WARMUP_ITERS:-2}"
VLLM_ITERS="${VLLM_ITERS:-5}"
VLLM_DECODE_PREFIX_TOKENS="${VLLM_DECODE_PREFIX_TOKENS:-256}"
LABEL=""

resolve_path() {
  python3 - "$1" <<'PY'
import pathlib
import sys

print(pathlib.Path(sys.argv[1]).resolve())
PY
}

resolve_repo_relative_path() {
  python3 - "$1" "$2" <<'PY'
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
candidate = pathlib.Path(sys.argv[2])
if not candidate.is_absolute():
    candidate = repo_root / candidate
print(candidate.resolve())
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --bench-build-dir)
      BENCH_BUILD_DIR="$2"
      shift 2
      ;;
    --test-build-dir)
      TEST_BUILD_DIR="$2"
      shift 2
      ;;
    --artifact-dir)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --decode-tokens)
      DECODE_TOKENS="$2"
      shift 2
      ;;
    --prefix-warmup)
      PREFIX_WARMUP="$2"
      shift 2
      ;;
    --prefix-iters)
      PREFIX_ITERS="$2"
      shift 2
      ;;
    --vllm-warmup-iters)
      VLLM_WARMUP_ITERS="$2"
      shift 2
      ;;
    --vllm-iters)
      VLLM_ITERS="$2"
      shift 2
      ;;
    --vllm-decode-prefix)
      VLLM_DECODE_PREFIX_TOKENS="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "bench_full_comparison.sh: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "bench_full_comparison.sh: ${name} must be a positive integer" >&2
    exit 1
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "bench_full_comparison.sh: ${name} must be a non-negative integer" >&2
    exit 1
  fi
}

require_positive_int "--decode-tokens" "${DECODE_TOKENS}"
require_nonnegative_int "--prefix-warmup" "${PREFIX_WARMUP}"
require_positive_int "--prefix-iters" "${PREFIX_ITERS}"
require_nonnegative_int "--vllm-warmup-iters" "${VLLM_WARMUP_ITERS}"
require_positive_int "--vllm-iters" "${VLLM_ITERS}"
require_positive_int "--vllm-decode-prefix" "${VLLM_DECODE_PREFIX_TOKENS}"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

if [[ -z "${BENCH_BUILD_DIR}" ]]; then
  for candidate in build build-benchmarks build-phase1 build-phase1-tests; do
    if [[ -x "${REPO_ROOT}/${candidate}/benchmarks/nano_fused_decode/nano_fused_decode_bench" ]] &&
       [[ -x "${REPO_ROOT}/${candidate}/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench" ]]; then
      BENCH_BUILD_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${TEST_BUILD_DIR}" ]]; then
  for candidate in build build-phase1-tests build-phase1 build-benchmarks; do
    if [[ -x "${REPO_ROOT}/${candidate}/testing/nano_save_prompt_oracle" ]]; then
      TEST_BUILD_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${BENCH_BUILD_DIR}" ]]; then
  echo "bench_full_comparison.sh: failed to locate benchmark binaries; set --bench-build-dir" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "bench_full_comparison.sh: manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "bench_full_comparison.sh: python3 is required" >&2
  exit 1
fi

REPO_ROOT="$(resolve_path "${REPO_ROOT}")"
SCRIPT_DIR="$(resolve_path "${SCRIPT_DIR}")"
ARTIFACT_ROOT="$(resolve_repo_relative_path "${REPO_ROOT}" "${ARTIFACT_ROOT}")"
MANIFEST_PATH="$(resolve_path "${MANIFEST_PATH}")"
BENCH_BUILD_DIR_PATH="$(resolve_repo_relative_path "${REPO_ROOT}" "${BENCH_BUILD_DIR}")"
TEST_BUILD_DIR_PATH=""
if [[ -n "${TEST_BUILD_DIR}" ]]; then
  TEST_BUILD_DIR_PATH="$(resolve_repo_relative_path "${REPO_ROOT}" "${TEST_BUILD_DIR}")"
fi

DECODE_BINARY="${BENCH_BUILD_DIR_PATH}/benchmarks/nano_fused_decode/nano_fused_decode_bench"
PREFIX_BINARY="${BENCH_BUILD_DIR_PATH}/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench"
VLLM_WRAPPER="${SCRIPT_DIR}/run_bench_vllm.sh"
DEFAULT_VLLM_PYTHON_BIN="${REPO_ROOT}/vllm-env-cu128/bin/python"
VLLM_PYTHON_BIN="${VLLM_PYTHON:-${DEFAULT_VLLM_PYTHON_BIN}}"
VLLM_SOURCE_TREE="${REPO_ROOT}/third_party/vllm"
VLLM_SOURCE_TREE="$(resolve_path "${VLLM_SOURCE_TREE}")"
REPO_GIT_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
VLLM_GIT_REVISION="$(git -C "${VLLM_SOURCE_TREE}" rev-parse HEAD 2>/dev/null || true)"
VLLM_GIT_TAG="$(git -C "${VLLM_SOURCE_TREE}" describe --tags --always 2>/dev/null || true)"
MODEL_ID="$(python3 - "${MANIFEST_PATH}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["runtime"]["model_id"])
PY
)"

if [[ ! -x "${DECODE_BINARY}" ]]; then
  echo "bench_full_comparison.sh: decode benchmark binary not executable: ${DECODE_BINARY}" >&2
  exit 1
fi
if [[ ! -x "${PREFIX_BINARY}" ]]; then
  echo "bench_full_comparison.sh: prefix benchmark binary not executable: ${PREFIX_BINARY}" >&2
  exit 1
fi
if [[ ! -f "${VLLM_WRAPPER}" ]]; then
  echo "bench_full_comparison.sh: vLLM wrapper not found: ${VLLM_WRAPPER}" >&2
  exit 1
fi

ORACLE_BINARY=""
if [[ -n "${TEST_BUILD_DIR_PATH}" ]] && [[ -x "${TEST_BUILD_DIR_PATH}/testing/nano_save_prompt_oracle" ]]; then
  ORACLE_BINARY="${TEST_BUILD_DIR_PATH}/testing/nano_save_prompt_oracle"
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}${LABEL:+_${LABEL}}"
mkdir -p "${RUN_DIR}/internal" "${RUN_DIR}/vllm" "${RUN_DIR}/fallback"

collect_runtime_backend_env_json() {
  python3 - "$@" <<'PY'
import json
import os
import sys

payload = {
    key: value
    for key, value in os.environ.items()
    if key.startswith("NEMOTRON_FORWARD_") and value
}
for raw in sys.argv[1:]:
    if "=" not in raw:
        continue
    key, value = raw.split("=", 1)
    if key.startswith("NEMOTRON_FORWARD_") and value:
        payload[key] = value
print(json.dumps(dict(sorted(payload.items())), sort_keys=True))
PY
}

VLLM_PYTHONPATH_VALUE="${REPO_ROOT}/third_party/vllm${PYTHONPATH:+:${PYTHONPATH}}"
VLLM_PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
VLLM_USE_FLASHINFER_MOE_FP4_VALUE="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
VLLM_FLASHINFER_MOE_BACKEND_VALUE="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}"
VLLM_ALLOW_INSECURE_SERIALIZATION_VALUE="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
VLLM_NVFP4_GEMM_BACKEND_VALUE="${VLLM_NVFP4_GEMM_BACKEND:-}"
VLLM_MOE_PADDING_VALUE="${VLLM_MOE_PADDING:-}"
VLLM_BACKEND_ENV_JSON="$(python3 - "${VLLM_PYTHONPATH_VALUE}" "${VLLM_PYTORCH_CUDA_ALLOC_CONF_VALUE}" "${VLLM_ALLOW_INSECURE_SERIALIZATION_VALUE}" "${VLLM_FLASHINFER_MOE_BACKEND_VALUE}" "${VLLM_USE_FLASHINFER_MOE_FP4_VALUE}" "${VLLM_NVFP4_GEMM_BACKEND_VALUE}" "${VLLM_MOE_PADDING_VALUE}" <<'PY'
import json
import sys

payload = {
    "PYTHONPATH": sys.argv[1],
    "PYTORCH_CUDA_ALLOC_CONF": sys.argv[2],
    "VLLM_ALLOW_INSECURE_SERIALIZATION": sys.argv[3],
    "VLLM_FLASHINFER_MOE_BACKEND": sys.argv[4],
    "VLLM_USE_FLASHINFER_MOE_FP4": sys.argv[5],
}
if sys.argv[6]:
    payload["VLLM_NVFP4_GEMM_BACKEND"] = sys.argv[6]
if sys.argv[7]:
    payload["VLLM_MOE_PADDING"] = sys.argv[7]
print(json.dumps(dict(sorted(payload.items())), sort_keys=True))
PY
)"

augment_json_artifact() {
  local artifact_path="$1"
  local artifact_family="$2"
  local binary_path="$3"
  local build_dir="$4"
  local manifest_path="$5"
  local model_id="$6"
  local backend_env_json="$7"
  python3 - "${artifact_path}" "${artifact_family}" "${binary_path}" "${build_dir}" "${manifest_path}" "${model_id}" "${backend_env_json}" "${REPO_ROOT}" "${VLLM_WRAPPER}" "${VLLM_PYTHON_BIN}" "${VLLM_SOURCE_TREE}" "${VLLM_GIT_REVISION}" "${VLLM_GIT_TAG}" <<'PY'
import json
import pathlib
import sys

artifact_path = pathlib.Path(sys.argv[1])
payload = json.loads(artifact_path.read_text(encoding="utf-8"))
backend_env = json.loads(sys.argv[7])

def null_if_empty(value: str):
    return value if value else None

payload["provenance"] = {
    "artifact_family": sys.argv[2],
    "repo_root": sys.argv[8],
    "binary_path": null_if_empty(sys.argv[3]),
    "build_dir": null_if_empty(sys.argv[4]),
    "manifest_path": null_if_empty(sys.argv[5]),
    "model_id": null_if_empty(sys.argv[6]),
    "backend_selection_env": backend_env,
    "external_vllm_reference": {
        "wrapper_path": sys.argv[9],
        "python_bin": sys.argv[10],
        "source_tree": sys.argv[11],
        "git_revision": null_if_empty(sys.argv[12]),
        "git_tag": null_if_empty(sys.argv[13]),
    },
}
artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=${REPO_ROOT}"
  echo "bench_build_dir=${BENCH_BUILD_DIR_PATH}"
  echo "test_build_dir=${TEST_BUILD_DIR_PATH:-missing}"
  echo "decode_binary=${DECODE_BINARY}"
  echo "prefix_binary=${PREFIX_BINARY}"
  echo "oracle_binary=${ORACLE_BINARY:-missing}"
  echo "manifest=${MANIFEST_PATH}"
  echo "model_id=${MODEL_ID}"
  echo "artifact_dir=${RUN_DIR}"
  echo "decode_tokens=${DECODE_TOKENS}"
  echo "prefix_warmup=${PREFIX_WARMUP}"
  echo "prefix_iters=${PREFIX_ITERS}"
  echo "vllm_warmup_iters=${VLLM_WARMUP_ITERS}"
  echo "vllm_iters=${VLLM_ITERS}"
  echo "vllm_decode_prefix_tokens=${VLLM_DECODE_PREFIX_TOKENS}"
  echo "vllm_python=${VLLM_PYTHON_BIN}"
  echo "vllm_source_tree=${VLLM_SOURCE_TREE}"
  echo "vllm_git_revision=${VLLM_GIT_REVISION}"
  echo "vllm_git_tag=${VLLM_GIT_TAG}"
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
  echo "[env]"
  env | grep -E '^(NEMOTRON_|VLLM_|CUDA_VISIBLE_DEVICES=|LD_LIBRARY_PATH=|PYTHONPATH=)' | sort || true
  echo
  echo "[vllm-backend-env-json]"
  echo "${VLLM_BACKEND_ENV_JSON}"
} > "${RUN_DIR}/environment.txt"

parse_prefix_stdout_to_json() {
  local stdout_path="$1"
  local json_path="$2"
  local config_name="$3"
  python3 - "${stdout_path}" "${json_path}" "${config_name}" <<'PY'
import json
import pathlib
import re
import sys

stdout_path = pathlib.Path(sys.argv[1])
json_path = pathlib.Path(sys.argv[2])
config_name = sys.argv[3]

header_pattern = re.compile(
    r"^nano_prefix_cache_ttft_bench: manifest=(?P<manifest>\S+) "
    r"model_id=(?P<model_id>\S+) "
    r"warmup_iterations=(?P<warmup>\d+) measured_iterations=(?P<iters>\d+) "
    r"tail_token_count=(?P<tail>\d+) "
    r"moe_prefill_window_tokens=(?P<window>\d+)$"
)
case_pattern = re.compile(
    r"^Case: (?P<name>\S+) \(total_prompt_tokens=(?P<total>\d+)\)$"
)
metric_pattern = re.compile(r"^\s{2}(?P<label>[^:]+): (?P<body>.+)$")
value_pattern = re.compile(
    r"median=(?P<median>[-+0-9.eE]+)\s*(?P<median_unit>[A-Za-z+]*)\s+"
    r"p95=(?P<p95>[-+0-9.eE]+)\s*(?P<p95_unit>[A-Za-z+]*)$"
)

payload = {
    "benchmark_name": "nano_prefix_cache_ttft_bench",
    "config_name": config_name,
    "stdout_path": str(stdout_path),
    "status": "ok",
    "cases": {},
}

current_case = None
for raw_line in stdout_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.rstrip("\n")
    header_match = header_pattern.match(line)
    if header_match:
        payload["manifest_path"] = header_match.group("manifest")
        payload["model_id"] = header_match.group("model_id")
        payload["warmup_iterations"] = int(header_match.group("warmup"))
        payload["measured_iterations"] = int(header_match.group("iters"))
        payload["tail_token_count"] = int(header_match.group("tail"))
        payload["moe_prefill_window_tokens"] = int(header_match.group("window"))
        continue

    case_match = case_pattern.match(line)
    if case_match:
        current_case = case_match.group("name")
        payload["cases"][current_case] = {
            "name": current_case,
            "total_prompt_tokens": int(case_match.group("total")),
            "metrics": {},
        }
        continue

    metric_match = metric_pattern.match(line)
    if metric_match and current_case is not None:
        label = metric_match.group("label").strip().lower().replace("/", "_").replace(" ", "_")
        body = metric_match.group("body").strip()
        if body == "n/a":
            payload["cases"][current_case]["metrics"][label] = None
            continue
        value_match = value_pattern.match(body)
        if not value_match:
            payload["cases"][current_case]["metrics"][label] = {
                "raw": body,
            }
            continue
        median_unit = value_match.group("median_unit")
        p95_unit = value_match.group("p95_unit")
        payload["cases"][current_case]["metrics"][label] = {
            "median": float(value_match.group("median")),
            "p95": float(value_match.group("p95")),
            "unit": median_unit or p95_unit or None,
            "raw": body,
        }

json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_decode_case() {
  local label="$1"
  shift
  local -a env_overrides=("$@")
  local json_path="${RUN_DIR}/internal/${label}.decode.json"
  local stdout_path="${RUN_DIR}/internal/${label}.decode.stdout.txt"
  local env_path="${RUN_DIR}/internal/${label}.decode.env.txt"
  local runtime_env_json
  runtime_env_json="$(collect_runtime_backend_env_json "NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}" "${env_overrides[@]}")"

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label=${label}"
    echo "binary=${DECODE_BINARY}"
    echo "build_dir=${BENCH_BUILD_DIR_PATH}"
    echo "manifest=${MANIFEST_PATH}"
    echo "model_id=${MODEL_ID}"
    echo "decode_tokens=${DECODE_TOKENS}"
    echo
    echo "[env-overrides]"
    printf '%s\n' "${env_overrides[@]}"
    echo
    echo "[backend-selection-env-json]"
    echo "${runtime_env_json}"
  } > "${env_path}"

  local -a cmd=(env)
  cmd+=("NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}")
  cmd+=("${env_overrides[@]}")
  cmd+=(
    "${DECODE_BINARY}"
    "--manifest" "${MANIFEST_PATH}"
    "--mode=steady-state"
    "--decode-tokens" "${DECODE_TOKENS}"
    "--json-output" "${json_path}"
  )

  echo "bench_full_comparison.sh: running decode case ${label}"
  "${cmd[@]}" 2>&1 | tee "${stdout_path}"
  augment_json_artifact \
    "${json_path}" \
    "runtime_decode" \
    "${DECODE_BINARY}" \
    "${BENCH_BUILD_DIR_PATH}" \
    "${MANIFEST_PATH}" \
    "${MODEL_ID}" \
    "${runtime_env_json}"
}

run_prefix_case() {
  local label="$1"
  local tail_token_count="$2"
  local moe_prefill_window_tokens="$3"
  shift 3
  local -a env_overrides=("$@")
  local stdout_path="${RUN_DIR}/internal/${label}.prefix.stdout.txt"
  local json_path="${RUN_DIR}/internal/${label}.prefix.json"
  local env_path="${RUN_DIR}/internal/${label}.prefix.env.txt"
  local runtime_env_json
  runtime_env_json="$(collect_runtime_backend_env_json "NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}" "${env_overrides[@]}")"

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label=${label}"
    echo "binary=${PREFIX_BINARY}"
    echo "build_dir=${BENCH_BUILD_DIR_PATH}"
    echo "manifest=${MANIFEST_PATH}"
    echo "model_id=${MODEL_ID}"
    echo "warmup=${PREFIX_WARMUP}"
    echo "iterations=${PREFIX_ITERS}"
    echo "tail_token_count=${tail_token_count}"
    echo "moe_prefill_window_tokens=${moe_prefill_window_tokens}"
    echo
    echo "[env-overrides]"
    printf '%s\n' "${env_overrides[@]}"
    echo
    echo "[backend-selection-env-json]"
    echo "${runtime_env_json}"
  } > "${env_path}"

  local -a cmd=(env)
  cmd+=("NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}")
  cmd+=("${env_overrides[@]}")
  cmd+=(
    "${PREFIX_BINARY}"
    "--warmup" "${PREFIX_WARMUP}"
    "--iterations" "${PREFIX_ITERS}"
    "--tail-token-count" "${tail_token_count}"
    "--moe-prefill-window-tokens" "${moe_prefill_window_tokens}"
  )

  echo "bench_full_comparison.sh: running prefix case ${label}"
  "${cmd[@]}" 2>&1 | tee "${stdout_path}"
  parse_prefix_stdout_to_json "${stdout_path}" "${json_path}" "${label}"
  augment_json_artifact \
    "${json_path}" \
    "runtime_prefix_ttft" \
    "${PREFIX_BINARY}" \
    "${BENCH_BUILD_DIR_PATH}" \
    "${MANIFEST_PATH}" \
    "${MODEL_ID}" \
    "${runtime_env_json}"
}

run_oracle_case() {
  local label="$1"
  shift
  local -a env_overrides=("$@")
  local json_path="${RUN_DIR}/fallback/${label}.oracle.json"
  local stdout_path="${RUN_DIR}/fallback/${label}.oracle.stdout.txt"
  local env_path="${RUN_DIR}/fallback/${label}.oracle.env.txt"
  local runtime_env_json
  runtime_env_json="$(collect_runtime_backend_env_json "NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}" "${env_overrides[@]}")"

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label=${label}"
    echo "binary=${ORACLE_BINARY}"
    echo "build_dir=${TEST_BUILD_DIR_PATH:-missing}"
    echo "manifest=${MANIFEST_PATH}"
    echo "model_id=${MODEL_ID}"
    echo
    echo "[env-overrides]"
    printf '%s\n' "${env_overrides[@]}"
    echo
    echo "[backend-selection-env-json]"
    echo "${runtime_env_json}"
  } > "${env_path}"

  local -a cmd=(env)
  cmd+=("NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}")
  cmd+=("NEMOTRON_REPO_ROOT=${REPO_ROOT}")
  cmd+=("NEMOTRON_GIT_REVISION=${REPO_GIT_REVISION}")
  if [[ -n "${TEST_BUILD_DIR_PATH}" ]]; then
    cmd+=("NEMOTRON_ORACLE_BUILD_DIR=${TEST_BUILD_DIR_PATH}")
  fi
  cmd+=("${env_overrides[@]}")
  cmd+=("${ORACLE_BINARY}" "--output" "${json_path}")

  echo "bench_full_comparison.sh: running fallback oracle case ${label}"
  "${cmd[@]}" 2>&1 | tee "${stdout_path}"
  augment_json_artifact \
    "${json_path}" \
    "runtime_oracle" \
    "${ORACLE_BINARY}" \
    "${TEST_BUILD_DIR_PATH:-}" \
    "${MANIFEST_PATH}" \
    "${MODEL_ID}" \
    "${runtime_env_json}"
}

compare_oracles() {
  local baseline_json="$1"
  local candidate_json="$2"
  local output_json="$3"
  local output_txt="$4"
  python3 - "${baseline_json}" "${candidate_json}" "${output_json}" "${output_txt}" <<'PY'
import json
import pathlib
import sys

baseline_path = pathlib.Path(sys.argv[1])
candidate_path = pathlib.Path(sys.argv[2])
output_json_path = pathlib.Path(sys.argv[3])
output_txt_path = pathlib.Path(sys.argv[4])

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

baseline_top5 = baseline.get("boundary_top5", [])
candidate_top5 = candidate.get("boundary_top5", [])

index_match = [item.get("index") for item in baseline_top5] == [
    item.get("index") for item in candidate_top5
]
value_diffs = []
for left, right in zip(baseline_top5, candidate_top5):
    value_diffs.append(abs(float(left["value"]) - float(right["value"])))

generated_tokens_match = baseline.get("generated_token_ids") == candidate.get("generated_token_ids")
boundary_token_match = baseline.get("boundary_token_id") == candidate.get("boundary_token_id")
max_abs_diff = max(value_diffs) if value_diffs else None

payload = {
    "baseline_json": str(baseline_path),
    "candidate_json": str(candidate_path),
    "generated_tokens_match": generated_tokens_match,
    "boundary_token_match": boundary_token_match,
    "boundary_top5_index_match": index_match,
    "boundary_top5_max_abs_diff": max_abs_diff,
    "pass": generated_tokens_match and boundary_token_match and index_match and (max_abs_diff == 0.0),
}

lines = [
    "Fallback disable check",
    f"baseline={baseline_path}",
    f"candidate={candidate_path}",
    f"generated_tokens_match={generated_tokens_match}",
    f"boundary_token_match={boundary_token_match}",
    f"boundary_top5_index_match={index_match}",
    f"boundary_top5_max_abs_diff={max_abs_diff}",
    f"pass={payload['pass']}",
]

output_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
output_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

run_decode_case \
  default

run_decode_case \
  unified_fused \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1

run_decode_case \
  fallback_disable_override \
  NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1

run_prefix_case \
  default \
  32 \
  32

run_prefix_case \
  unified_fused \
  32 \
  32 \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1

run_prefix_case \
  unified_fused_moe_window64 \
  64 \
  64 \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1

run_prefix_case \
  unified_fused_moe_window128 \
  128 \
  128 \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1

VLLM_JSON_PATH="${RUN_DIR}/vllm/vllm_flashinfer_cutlass.json"
VLLM_STDOUT_PATH="${RUN_DIR}/vllm/vllm_flashinfer_cutlass.stdout.txt"
echo "bench_full_comparison.sh: running vLLM flashinfer_cutlass matrix"
bash "${VLLM_WRAPPER}" \
  --output "${VLLM_JSON_PATH}" \
  --token-counts 1 32 64 128 \
  --warmup-iters "${VLLM_WARMUP_ITERS}" \
  --iters "${VLLM_ITERS}" \
  --decode-prefix-tokens "${VLLM_DECODE_PREFIX_TOKENS}" \
  2>&1 | tee "${VLLM_STDOUT_PATH}"
augment_json_artifact \
  "${VLLM_JSON_PATH}" \
  "external_vllm_ttft" \
  "${VLLM_PYTHON_BIN}" \
  "" \
  "" \
  "${MODEL_ID}" \
  "${VLLM_BACKEND_ENV_JSON}"

if [[ -n "${ORACLE_BINARY}" ]]; then
  run_oracle_case \
    default \
    NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
    NEMOTRON_FORWARD_FUSED_MOE_PREFILL=0

  run_oracle_case \
    fallback_disable_override \
    NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
    NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1

  compare_oracles \
    "${RUN_DIR}/fallback/default.oracle.json" \
    "${RUN_DIR}/fallback/fallback_disable_override.oracle.json" \
    "${RUN_DIR}/fallback/fallback_disable_check.json" \
    "${RUN_DIR}/fallback/fallback_disable_check.txt"
  augment_json_artifact \
    "${RUN_DIR}/fallback/fallback_disable_check.json" \
    "runtime_fallback_check" \
    "${ORACLE_BINARY}" \
    "${TEST_BUILD_DIR_PATH:-}" \
    "${MANIFEST_PATH}" \
    "${MODEL_ID}" \
    "$(collect_runtime_backend_env_json "NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}")"
fi

python3 - "${RUN_DIR}" <<'PY'
import json
import math
import pathlib
import statistics
import sys

run_dir = pathlib.Path(sys.argv[1])

def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

def format_metric(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.6f}"

rows = []

decode_case_specs = [
    ("default", run_dir / "internal" / "default.decode.json"),
    ("unified_fused", run_dir / "internal" / "unified_fused.decode.json"),
    ("fallback_disable_override", run_dir / "internal" / "fallback_disable_override.decode.json"),
]
for config_name, path in decode_case_specs:
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark = payload["benchmark"]
    step_ms = [float(value) for value in payload.get("steady_state_step_ms", [])]
    rows.append(
        {
            "family": "runtime",
            "config": config_name,
            "workload": "decode_1",
            "metric": "steady_state_decode_step_ms",
            "median": statistics.median(step_ms) if step_ms else float(benchmark["hot_steady_state_mean_ms"]),
            "p95": percentile(step_ms, 0.95) if step_ms else float(benchmark["hot_steady_state_mean_ms"]),
            "status": "ok",
            "artifact": str(path),
        }
    )

prefix_case_specs = [
    ("default", run_dir / "internal" / "default.prefix.json"),
    ("unified_fused", run_dir / "internal" / "unified_fused.prefix.json"),
    ("unified_fused_moe_window64", run_dir / "internal" / "unified_fused_moe_window64.prefix.json"),
    ("unified_fused_moe_window128", run_dir / "internal" / "unified_fused_moe_window128.prefix.json"),
]
for config_name, path in prefix_case_specs:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for case_name, case_payload in sorted(payload["cases"].items()):
        if "_tail" not in case_name:
            continue
        metrics = case_payload["metrics"]
        tail_prefill = metrics.get("tail_prefill_latency")
        hot_ttft = metrics.get("hot-prefix_ttft")
        if tail_prefill is not None:
            rows.append(
                {
                    "family": "runtime",
                    "config": config_name,
                    "workload": case_name,
                    "metric": "tail_prefill_latency_ms",
                    "median": float(tail_prefill["median"]),
                    "p95": float(tail_prefill["p95"]),
                    "status": "ok",
                    "artifact": str(path),
                }
            )
        if hot_ttft is not None:
            rows.append(
                {
                    "family": "runtime",
                    "config": config_name,
                    "workload": case_name,
                    "metric": "hot_prefix_ttft_ms",
                    "median": float(hot_ttft["median"]),
                    "p95": float(hot_ttft["p95"]),
                    "status": "ok",
                    "artifact": str(path),
                }
            )

vllm_payload = json.loads((run_dir / "vllm" / "vllm_flashinfer_cutlass.json").read_text(encoding="utf-8"))
for case in vllm_payload.get("cases", []):
    label = str(case["label"])
    summary = case["ttft_summary_ms"]
    rows.append(
        {
            "family": "vllm",
            "config": "flashinfer_cutlass",
            "workload": label,
            "metric": "ttft_ms",
            "median": float(summary["median_ms"]),
            "p95": float(summary["p95_ms"]),
            "status": "ok",
            "artifact": str(run_dir / "vllm" / "vllm_flashinfer_cutlass.json"),
        }
    )

fallback_path = run_dir / "fallback" / "fallback_disable_check.json"
fallback_payload = None
if fallback_path.exists():
    fallback_payload = json.loads(fallback_path.read_text(encoding="utf-8"))

summary_json = {
    "run_dir": str(run_dir),
    "rows": rows,
    "fallback_disable_check": fallback_payload,
    "notes": [
        "Decode metrics come from nano_fused_decode_bench steady-state per-step timings.",
        "Cached-tail metrics come from parsed nano_prefix_cache_ttft_bench stdout summaries.",
        "Unified-fused prefix runs explicitly sweep 32/64/128-token tail windows via nano_prefix_cache_ttft_bench CLI controls.",
    ],
}
summary_json_path = run_dir / "summary.json"
summary_json_path.write_text(json.dumps(summary_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = []
lines.append("Full comparison matrix")
lines.append("")
lines.append(
    f"{'family':<8} {'config':<28} {'workload':<40} {'metric':<28} "
    f"{'median':>12} {'p95':>12} {'status':<12} artifact"
)
for row in rows:
    lines.append(
        f"{row['family']:<8} {row['config']:<28} {row['workload']:<40} {row['metric']:<28} "
        f"{format_metric(row['median']):>12} {format_metric(row['p95']):>12} "
        f"{row['status']:<12} {row['artifact']}"
    )
    note = row.get("note")
    if note:
        lines.append(f"note: {note}")

lines.append("")
lines.append("Fallback disable check")
if fallback_payload is None:
    lines.append("status=unavailable reason=nano_save_prompt_oracle_not_found")
else:
    lines.append(
        " ".join(
            [
                f"pass={fallback_payload['pass']}",
                f"generated_tokens_match={fallback_payload['generated_tokens_match']}",
                f"boundary_token_match={fallback_payload['boundary_token_match']}",
                f"boundary_top5_index_match={fallback_payload['boundary_top5_index_match']}",
                f"boundary_top5_max_abs_diff={fallback_payload['boundary_top5_max_abs_diff']}",
            ]
        )
    )

summary_txt_path = run_dir / "summary.txt"
summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

augment_json_artifact \
  "${RUN_DIR}/summary.json" \
  "full_comparison_summary" \
  "${SCRIPT_DIR}/bench_full_comparison.sh" \
  "" \
  "${MANIFEST_PATH}" \
  "${MODEL_ID}" \
  "$(collect_runtime_backend_env_json "NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}")"

echo "bench_full_comparison.sh: artifacts written to ${RUN_DIR}"
