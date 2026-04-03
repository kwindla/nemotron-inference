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
  - The current nano_prefix_cache_ttft_bench executable hardcodes 32-token tail cases
    and does not expose moe_prefill_window_tokens. This harness therefore emits
    structured "unavailable" JSON records for the internal 64/128 chunk-window rows
    instead of fabricating numbers.
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

DECODE_BINARY="${REPO_ROOT}/${BENCH_BUILD_DIR}/benchmarks/nano_fused_decode/nano_fused_decode_bench"
PREFIX_BINARY="${REPO_ROOT}/${BENCH_BUILD_DIR}/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench"
VLLM_WRAPPER="${SCRIPT_DIR}/run_bench_vllm.sh"

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
if [[ -n "${TEST_BUILD_DIR}" ]] && [[ -x "${REPO_ROOT}/${TEST_BUILD_DIR}/testing/nano_save_prompt_oracle" ]]; then
  ORACLE_BINARY="${REPO_ROOT}/${TEST_BUILD_DIR}/testing/nano_save_prompt_oracle"
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}${LABEL:+_${LABEL}}"
mkdir -p "${RUN_DIR}/internal" "${RUN_DIR}/vllm" "${RUN_DIR}/fallback"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=${REPO_ROOT}"
  echo "bench_build_dir=${BENCH_BUILD_DIR}"
  echo "test_build_dir=${TEST_BUILD_DIR:-missing}"
  echo "decode_binary=${DECODE_BINARY}"
  echo "prefix_binary=${PREFIX_BINARY}"
  echo "oracle_binary=${ORACLE_BINARY:-missing}"
  echo "manifest=${MANIFEST_PATH}"
  echo "artifact_dir=${RUN_DIR}"
  echo "decode_tokens=${DECODE_TOKENS}"
  echo "prefix_warmup=${PREFIX_WARMUP}"
  echo "prefix_iters=${PREFIX_ITERS}"
  echo "vllm_warmup_iters=${VLLM_WARMUP_ITERS}"
  echo "vllm_iters=${VLLM_ITERS}"
  echo "vllm_decode_prefix_tokens=${VLLM_DECODE_PREFIX_TOKENS}"
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
} > "${RUN_DIR}/environment.txt"

write_unavailable_json() {
  local output_path="$1"
  local config_name="$2"
  local workload="$3"
  local requested_window="$4"
  python3 - "${output_path}" "${config_name}" "${workload}" "${requested_window}" <<'PY'
import json
import pathlib
import sys

output_path = pathlib.Path(sys.argv[1])
config_name = sys.argv[2]
workload = sys.argv[3]
requested_window = int(sys.argv[4])

payload = {
    "status": "unavailable",
    "config_name": config_name,
    "workload": workload,
    "requested_moe_prefill_window_tokens": requested_window,
    "reason": (
        "The current nano_prefix_cache_ttft_bench executable hardcodes 32-token tail "
        "cases and does not expose moe_prefill_window_tokens."
    ),
    "required_surface": (
        "A benchmark executable or wrapper that can drive the existing runtime with "
        "SingleTokenForwardConfig.moe_prefill_window_tokens=%d using the existing "
        "unified backend implementation." % requested_window
    ),
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

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
    r"warmup_iterations=(?P<warmup>\d+) measured_iterations=(?P<iters>\d+)$"
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
        payload["warmup_iterations"] = int(header_match.group("warmup"))
        payload["measured_iterations"] = int(header_match.group("iters"))
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

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label=${label}"
    echo "binary=${DECODE_BINARY}"
    echo "manifest=${MANIFEST_PATH}"
    echo "decode_tokens=${DECODE_TOKENS}"
    echo
    echo "[env-overrides]"
    printf '%s\n' "${env_overrides[@]}"
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
}

run_prefix_case() {
  local label="$1"
  shift
  local -a env_overrides=("$@")
  local stdout_path="${RUN_DIR}/internal/${label}.prefix.stdout.txt"
  local json_path="${RUN_DIR}/internal/${label}.prefix.json"
  local env_path="${RUN_DIR}/internal/${label}.prefix.env.txt"

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label=${label}"
    echo "binary=${PREFIX_BINARY}"
    echo "manifest=${MANIFEST_PATH}"
    echo "warmup=${PREFIX_WARMUP}"
    echo "iterations=${PREFIX_ITERS}"
    echo
    echo "[env-overrides]"
    printf '%s\n' "${env_overrides[@]}"
  } > "${env_path}"

  local -a cmd=(env)
  cmd+=("NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}")
  cmd+=("${env_overrides[@]}")
  cmd+=(
    "${PREFIX_BINARY}"
    "--warmup" "${PREFIX_WARMUP}"
    "--iterations" "${PREFIX_ITERS}"
  )

  echo "bench_full_comparison.sh: running prefix case ${label}"
  "${cmd[@]}" 2>&1 | tee "${stdout_path}"
  parse_prefix_stdout_to_json "${stdout_path}" "${json_path}" "${label}"
}

run_oracle_case() {
  local label="$1"
  shift
  local -a env_overrides=("$@")
  local json_path="${RUN_DIR}/fallback/${label}.oracle.json"
  local stdout_path="${RUN_DIR}/fallback/${label}.oracle.stdout.txt"
  local env_path="${RUN_DIR}/fallback/${label}.oracle.env.txt"

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label=${label}"
    echo "binary=${ORACLE_BINARY}"
    echo "manifest=${MANIFEST_PATH}"
    echo
    echo "[env-overrides]"
    printf '%s\n' "${env_overrides[@]}"
  } > "${env_path}"

  local -a cmd=(env)
  cmd+=("NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}")
  cmd+=("${env_overrides[@]}")
  cmd+=("${ORACLE_BINARY}" "--output" "${json_path}")

  echo "bench_full_comparison.sh: running fallback oracle case ${label}"
  "${cmd[@]}" 2>&1 | tee "${stdout_path}"
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
  default \
  NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=0 \
  NEMOTRON_FORWARD_MOE_CUBLASLT=0

run_decode_case \
  unified_fused \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1 \
  NEMOTRON_FORWARD_MOE_CUBLASLT=1

run_decode_case \
  fallback_disable_override \
  NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1 \
  NEMOTRON_FORWARD_MOE_CUBLASLT=0

run_prefix_case \
  default \
  NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=0

run_prefix_case \
  unified_fused \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1

write_unavailable_json \
  "${RUN_DIR}/internal/unified_fused_moe_window64.prefix.json" \
  "unified_fused_moe_window64" \
  "prefill_chunk64" \
  "64"

write_unavailable_json \
  "${RUN_DIR}/internal/unified_fused_moe_window128.prefix.json" \
  "unified_fused_moe_window128" \
  "prefill_chunk128" \
  "128"

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
]
tail_case_names = [
    "cached_committed_head_prefix256_tail32",
    "cached_committed_head_prefix1024_tail32",
    "cached_committed_head_prefix4096_tail32",
    "cached_global_root_prefix256_tail32",
    "cached_global_root_prefix1024_tail32",
    "cached_global_root_prefix4096_tail32",
]
for config_name, path in prefix_case_specs:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for case_name in tail_case_names:
        case_payload = payload["cases"].get(case_name)
        if case_payload is None:
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

for unavailable_name in [
    "unified_fused_moe_window64.prefix.json",
    "unified_fused_moe_window128.prefix.json",
]:
    path = run_dir / "internal" / unavailable_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "family": "runtime",
            "config": payload["config_name"],
            "workload": payload["workload"],
            "metric": "hot_prefix_ttft_ms",
            "median": None,
            "p95": None,
            "status": payload["status"],
            "artifact": str(path),
            "note": payload["reason"],
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
        "Internal 64/128 chunk-window rows are marked unavailable because the current benchmark executable does not expose moe_prefill_window_tokens.",
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

echo "bench_full_comparison.sh: artifacts written to ${RUN_DIR}"
