#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bench_decode_backends.sh [options] [-- extra nano_fused_decode_bench args]

Compares the current resident decode backends with nano_fused_decode_bench
using token_count == 1 decode steps after a fixed prompt prefill.

Options:
  --manifest PATH       Manifest path. Defaults to NEMOTRON_FORWARD_MANIFEST or the
                        standard Nano RTX 5090 manifest artifact.
  --build-dir DIR       Build directory containing nano_fused_decode_bench.
                        Defaults to NEMOTRON_BUILD_DIR or the first known build dir.
  --artifact-dir DIR    Output directory root. Default:
                        proj-2026-04-03-0318/artifacts/decode_backends
  --mode MODE           steady-state, cached-head, or profile-ready.
                        Default: steady-state
  --decode-tokens N     Number of token_count == 1 decode steps for steady-state or
                        cached-head. Default: 16
  --label TEXT          Optional suffix for the artifact directory name.

Any arguments after `--` are forwarded to nano_fused_decode_bench.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MANIFEST="${REPO_ROOT}/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json"
MANIFEST_PATH="${NEMOTRON_FORWARD_MANIFEST:-${DEFAULT_MANIFEST}}"
BUILD_DIR="${NEMOTRON_BUILD_DIR:-}"
ARTIFACT_ROOT="${SCRIPT_DIR}/artifacts/decode_backends"
MODE="${MODE:-steady-state}"
DECODE_TOKENS="${DECODE_TOKENS:-16}"
LABEL=""
EXTRA_BENCH_ARGS=()

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
    --mode)
      MODE="$2"
      shift 2
      ;;
    --decode-tokens)
      DECODE_TOKENS="$2"
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
    --)
      shift
      EXTRA_BENCH_ARGS=("$@")
      break
      ;;
    *)
      echo "bench_decode_backends.sh: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${MODE}" in
  steady-state|cached-head|profile-ready)
    ;;
  *)
    echo "bench_decode_backends.sh: unsupported mode '${MODE}'" >&2
    usage
    exit 1
    ;;
esac

if ! [[ "${DECODE_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "bench_decode_backends.sh: --decode-tokens must be a positive integer" >&2
  exit 1
fi

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
elif [[ -d /usr/local/cuda-13.2/compat ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

if [[ -z "${BUILD_DIR}" ]]; then
  for candidate in build-benchmarks build-phase1-tests build-phase1 build; do
    if [[ -x "${REPO_ROOT}/${candidate}/benchmarks/nano_fused_decode/nano_fused_decode_bench" ]]; then
      BUILD_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${BUILD_DIR}" ]]; then
  echo "bench_decode_backends.sh: nano_fused_decode_bench not found; set --build-dir or NEMOTRON_BUILD_DIR" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "bench_decode_backends.sh: manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "bench_decode_backends.sh: python3 is required" >&2
  exit 1
fi

BINARY="${REPO_ROOT}/${BUILD_DIR}/benchmarks/nano_fused_decode/nano_fused_decode_bench"
if [[ ! -x "${BINARY}" ]]; then
  echo "bench_decode_backends.sh: benchmark binary not executable: ${BINARY}" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}${LABEL:+_${LABEL}}"
mkdir -p "${RUN_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "binary=${BINARY}"
  echo "manifest=${MANIFEST_PATH}"
  echo "mode=${MODE}"
  echo "decode_tokens=${DECODE_TOKENS}"
  echo "artifact_dir=${RUN_DIR}"
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
  echo "[forward-env]"
  env | grep -E '^(NEMOTRON_|CUDA_VISIBLE_DEVICES=|LD_LIBRARY_PATH=)' | sort || true
} > "${RUN_DIR}/environment.txt"

run_case() {
  local label="$1"
  shift
  local -a env_overrides=("$@")
  local json_path="${RUN_DIR}/${label}.json"
  local stdout_path="${RUN_DIR}/${label}.stdout.txt"
  local env_path="${RUN_DIR}/${label}.env.txt"

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label=${label}"
    echo "binary=${BINARY}"
    echo "manifest=${MANIFEST_PATH}"
    echo "mode=${MODE}"
    echo "decode_tokens=${DECODE_TOKENS}"
    echo
    echo "[env-overrides]"
    printf '%s\n' "${env_overrides[@]}"
  } > "${env_path}"

  local -a cmd=(env)
  cmd+=("${env_overrides[@]}")
  cmd+=("${BINARY}" "--manifest" "${MANIFEST_PATH}" "--mode=${MODE}")
  if [[ "${MODE}" == "steady-state" || "${MODE}" == "cached-head" ]]; then
    cmd+=("--decode-tokens" "${DECODE_TOKENS}")
  fi
  cmd+=("--json-output" "${json_path}")
  cmd+=("${EXTRA_BENCH_ARGS[@]}")

  echo "bench_decode_backends.sh: running ${label}"
  "${cmd[@]}" 2>&1 | tee "${stdout_path}"
}

run_case \
  default

run_case \
  unified_fused \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1

run_case \
  decode_cublaslt \
  NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=0

python3 - \
  "${RUN_DIR}/default.json" \
  "${RUN_DIR}/unified_fused.json" \
  "${RUN_DIR}/decode_cublaslt.json" \
  "${RUN_DIR}/summary.txt" <<'PY'
import json
import math
import pathlib
import sys

input_paths = [pathlib.Path(arg) for arg in sys.argv[1:4]]
summary_path = pathlib.Path(sys.argv[4])

rows = []
for path in input_paths:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    benchmark = payload["benchmark"]
    rows.append(
        {
            "label": path.stem,
            "mode": benchmark["mode"],
            "decode_tokens": benchmark["decode_token_count"],
            "decode_ms": float(benchmark["hot_steady_state_mean_ms"]),
            "prefill_ms": float(benchmark["steady_state_prefill_ms"]),
            "tokens_per_second": float(benchmark["steady_state_generated_tokens_per_second"]),
            "json_path": str(path),
        }
    )

baseline = next(row for row in rows if row["label"] == "default")
baseline_ms = baseline["decode_ms"]
best = min(rows, key=lambda row: row["decode_ms"])
unified = next(row for row in rows if row["label"] == "unified_fused")

lines = []
lines.append("Decode backend comparison")
lines.append(f"mode={baseline['mode']} decode_tokens={baseline['decode_tokens']}")
lines.append("")
for row in rows:
    delta_vs_scalar_pct = 0.0
    if baseline_ms > 0.0:
        delta_vs_scalar_pct = ((row["decode_ms"] / baseline_ms) - 1.0) * 100.0
    lines.append(
        f"{row['label']}: decode_mean_ms={row['decode_ms']:.6f} "
        f"delta_vs_default_pct={delta_vs_scalar_pct:+.2f} "
        f"tokens_per_second={row['tokens_per_second']:.3f} "
        f"prefill_ms={row['prefill_ms']:.6f} "
        f"json={row['json_path']}"
    )

lines.append("")
lines.append(f"winner={best['label']} decode_mean_ms={best['decode_ms']:.6f}")
delta_pct = 0.0
if baseline_ms > 0.0:
    delta_pct = abs(((unified["decode_ms"] / baseline_ms) - 1.0) * 100.0)
if delta_pct <= 2.0:
    lines.append("decision=default_matches_explicit_unified_within_noise")
elif unified["decode_ms"] <= baseline_ms:
    lines.append("decision=explicit_unified_beats_default_check_for_ambient_env_drift")
else:
    lines.append("decision=default_beats_explicit_unified_check_for_ambient_env_drift")

summary = "\n".join(lines) + "\n"
summary_path.write_text(summary, encoding="utf-8")
sys.stdout.write(summary)
PY

echo "bench_decode_backends.sh: artifacts written to ${RUN_DIR}"
