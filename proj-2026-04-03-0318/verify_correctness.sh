#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: verify_correctness.sh [options]

Run a short Nano inference under:
  - the current non-unified runtime path
  - the unified fused backend

Then compare the saved output oracles and run the existing nano_16_token_correctness_test
in oracle mode against the unified candidate.

Options:
  --manifest PATH          Manifest path. Defaults to NEMOTRON_FORWARD_MANIFEST or
                           artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json
  --build-dir DIR          Build directory containing nano_save_prompt_oracle and
                           nano_16_token_correctness_test. Defaults to auto-detect.
  --artifact-dir DIR       Output directory root. Default:
                           proj-2026-04-03-0318/artifacts/verify_correctness
  --atol FLOAT             Absolute tolerance for boundary top-5 logit comparison.
                           Default: 1e-3
  --label TEXT             Optional suffix for the artifact directory name.
  --help, -h               Show this message.

Exit status:
  0 if the oracle comparison and nano_16_token_correctness_test both pass
  1 otherwise

Note:
  The existing nano_save_prompt_oracle executable exposes the prompt-boundary top-5
  logits plus the generated token sequence, not the full dense logits tensor. This
  script therefore enforces numerical agreement on the saved top-5 values and exact
  agreement on the generated token sequence.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MANIFEST="${REPO_ROOT}/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json"
MANIFEST_PATH="${NEMOTRON_FORWARD_MANIFEST:-${DEFAULT_MANIFEST}}"
BUILD_DIR="${NEMOTRON_TEST_BUILD_DIR:-}"
ARTIFACT_ROOT="${SCRIPT_DIR}/artifacts/verify_correctness"
ATOL="${ATOL:-1e-3}"
LABEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --artifact-dir)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --atol)
      ATOL="$2"
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
      echo "verify_correctness.sh: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

if [[ -z "${BUILD_DIR}" ]]; then
  for candidate in build build-phase1-tests build-phase1 build-benchmarks; do
    if [[ -x "${REPO_ROOT}/${candidate}/testing/nano_save_prompt_oracle" ]] &&
       [[ -x "${REPO_ROOT}/${candidate}/testing/nano_16_token_correctness_test" ]]; then
      BUILD_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${BUILD_DIR}" ]]; then
  echo "verify_correctness.sh: failed to locate testing binaries; set --build-dir" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "verify_correctness.sh: manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "verify_correctness.sh: python3 is required" >&2
  exit 1
fi

SAVE_ORACLE_BINARY="${REPO_ROOT}/${BUILD_DIR}/testing/nano_save_prompt_oracle"
CORRECTNESS_BINARY="${REPO_ROOT}/${BUILD_DIR}/testing/nano_16_token_correctness_test"

if [[ ! -x "${SAVE_ORACLE_BINARY}" ]]; then
  echo "verify_correctness.sh: missing binary: ${SAVE_ORACLE_BINARY}" >&2
  exit 1
fi
if [[ ! -x "${CORRECTNESS_BINARY}" ]]; then
  echo "verify_correctness.sh: missing binary: ${CORRECTNESS_BINARY}" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}${LABEL:+_${LABEL}}"
mkdir -p "${RUN_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "save_oracle_binary=${SAVE_ORACLE_BINARY}"
  echo "correctness_binary=${CORRECTNESS_BINARY}"
  echo "manifest=${MANIFEST_PATH}"
  echo "artifact_dir=${RUN_DIR}"
  echo "atol=${ATOL}"
  echo "uname=$(uname -a)"
  echo
  echo "[env]"
  env | grep -E '^(NEMOTRON_|CUDA_VISIBLE_DEVICES=|LD_LIBRARY_PATH=)' | sort || true
} > "${RUN_DIR}/environment.txt"

run_oracle_case() {
  local label="$1"
  shift
  local -a env_overrides=("$@")
  local json_path="${RUN_DIR}/${label}.oracle.json"
  local stdout_path="${RUN_DIR}/${label}.oracle.stdout.txt"
  local env_path="${RUN_DIR}/${label}.oracle.env.txt"

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label=${label}"
    echo "binary=${SAVE_ORACLE_BINARY}"
    echo "manifest=${MANIFEST_PATH}"
    echo
    echo "[env-overrides]"
    printf '%s\n' "${env_overrides[@]}"
  } > "${env_path}"

  local -a cmd=(env)
  cmd+=("NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}")
  cmd+=("${env_overrides[@]}")
  cmd+=("${SAVE_ORACLE_BINARY}" "--output" "${json_path}")

  echo "verify_correctness.sh: running oracle case ${label}"
  "${cmd[@]}" 2>&1 | tee "${stdout_path}"
}

run_oracle_correctness_test() {
  local baseline_oracle_path="$1"
  local stdout_path="${RUN_DIR}/unified_vs_default.correctness.stdout.txt"
  local env_path="${RUN_DIR}/unified_vs_default.correctness.env.txt"

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "binary=${CORRECTNESS_BINARY}"
    echo "manifest=${MANIFEST_PATH}"
    echo "oracle=${baseline_oracle_path}"
    echo
    echo "[env-overrides]"
    printf '%s\n' \
      "NEMOTRON_FORWARD_UNIFIED_FUSED=1" \
      "NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1" \
      "NEMOTRON_NANO_16_ORACLE_PATH=${baseline_oracle_path}"
  } > "${env_path}"

  local -a cmd=(env)
  cmd+=("NEMOTRON_FORWARD_MANIFEST=${MANIFEST_PATH}")
  cmd+=("NEMOTRON_FORWARD_UNIFIED_FUSED=1")
  cmd+=("NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1")
  cmd+=("NEMOTRON_NANO_16_ORACLE_PATH=${baseline_oracle_path}")
  cmd+=("${CORRECTNESS_BINARY}")

  echo "verify_correctness.sh: running nano_16_token_correctness_test against baseline oracle"
  "${cmd[@]}" 2>&1 | tee "${stdout_path}"
}

run_oracle_case \
  default_backend \
  NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=0

run_oracle_case \
  unified_fused_backend \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1

comparison_status=0
if python3 - "${RUN_DIR}/default_backend.oracle.json" "${RUN_DIR}/unified_fused_backend.oracle.json" "${ATOL}" "${RUN_DIR}/comparison.json" "${RUN_DIR}/comparison.txt" <<'PY'
import json
import pathlib
import sys

baseline_path = pathlib.Path(sys.argv[1])
candidate_path = pathlib.Path(sys.argv[2])
atol = float(sys.argv[3])
comparison_json_path = pathlib.Path(sys.argv[4])
comparison_txt_path = pathlib.Path(sys.argv[5])

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

baseline_top5 = baseline.get("boundary_top5", [])
candidate_top5 = candidate.get("boundary_top5", [])

generated_tokens_match = baseline.get("generated_token_ids") == candidate.get("generated_token_ids")
boundary_token_match = baseline.get("boundary_token_id") == candidate.get("boundary_token_id")
top5_length_match = len(baseline_top5) == len(candidate_top5)
top5_index_match = [item.get("index") for item in baseline_top5] == [
    item.get("index") for item in candidate_top5
]
value_diffs = []
for left, right in zip(baseline_top5, candidate_top5):
    value_diffs.append(abs(float(left["value"]) - float(right["value"])))
top5_max_abs_diff = max(value_diffs) if value_diffs else None

boundary_max_logit_diff = abs(
    float(baseline.get("boundary_max_logit", 0.0)) - float(candidate.get("boundary_max_logit", 0.0))
)

passed = (
    generated_tokens_match
    and boundary_token_match
    and top5_length_match
    and top5_index_match
    and top5_max_abs_diff is not None
    and top5_max_abs_diff <= atol
)

payload = {
    "baseline_json": str(baseline_path),
    "candidate_json": str(candidate_path),
    "atol": atol,
    "generated_tokens_match": generated_tokens_match,
    "boundary_token_match": boundary_token_match,
    "boundary_top5_length_match": top5_length_match,
    "boundary_top5_index_match": top5_index_match,
    "boundary_top5_max_abs_diff": top5_max_abs_diff,
    "boundary_max_logit_diff": boundary_max_logit_diff,
    "pass": passed,
}

lines = [
    "Unified correctness comparison",
    f"baseline={baseline_path}",
    f"candidate={candidate_path}",
    f"atol={atol}",
    f"generated_tokens_match={generated_tokens_match}",
    f"boundary_token_match={boundary_token_match}",
    f"boundary_top5_length_match={top5_length_match}",
    f"boundary_top5_index_match={top5_index_match}",
    f"boundary_top5_max_abs_diff={top5_max_abs_diff}",
    f"boundary_max_logit_diff={boundary_max_logit_diff}",
    f"pass={passed}",
]

comparison_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
comparison_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

if not passed:
    raise SystemExit(1)
PY
then
  comparison_status=0
else
  comparison_status=$?
fi

correctness_status=0
if run_oracle_correctness_test "${RUN_DIR}/default_backend.oracle.json"; then
  correctness_status=0
else
  correctness_status=$?
fi

python3 - "${RUN_DIR}" "${comparison_status}" "${correctness_status}" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
comparison_status = int(sys.argv[2])
correctness_status = int(sys.argv[3])
comparison = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
summary_lines = [
    "Correctness verification",
    f"oracle_compare_pass={comparison['pass']}",
    f"oracle_compare_exit_status={comparison_status}",
    f"generated_tokens_match={comparison['generated_tokens_match']}",
    f"boundary_token_match={comparison['boundary_token_match']}",
    f"boundary_top5_index_match={comparison['boundary_top5_index_match']}",
    f"boundary_top5_max_abs_diff={comparison['boundary_top5_max_abs_diff']}",
    f"boundary_max_logit_diff={comparison['boundary_max_logit_diff']}",
    f"nano_16_token_correctness_test_pass={correctness_status == 0}",
    f"nano_16_token_correctness_test_exit_status={correctness_status}",
]
summary_path = run_dir / "summary.txt"
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print("\n".join(summary_lines))
PY

echo "verify_correctness.sh: artifacts written to ${RUN_DIR}"

if [[ "${comparison_status}" -ne 0 || "${correctness_status}" -ne 0 ]]; then
  exit 1
fi
