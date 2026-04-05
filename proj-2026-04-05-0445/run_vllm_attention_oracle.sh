#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/proj-2026-04-05-0445"
BENCH_DIR="${ROOT_DIR}/third_party/vllm/benchmarks/attention_benchmarks"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${PROJECT_DIR}/oracle_runs/${TIMESTAMP}"

mkdir -p "${OUT_DIR}"

export PYTHONPATH="${ROOT_DIR}/third_party/vllm${PYTHONPATH:+:${PYTHONPATH}}"

case "${MODE}" in
  smoke)
    CONFIGS=(
      "${PROJECT_DIR}/vllm_attention_oracle_smoke.yaml"
    )
    ;;
  single)
    CONFIGS=(
      "${PROJECT_DIR}/vllm_attention_oracle_single.yaml"
    )
    ;;
  concurrency4)
    CONFIGS=(
      "${PROJECT_DIR}/vllm_attention_oracle_concurrency4.yaml"
    )
    ;;
  full)
    CONFIGS=(
      "${PROJECT_DIR}/vllm_attention_oracle_single.yaml"
      "${PROJECT_DIR}/vllm_attention_oracle_concurrency4.yaml"
    )
    ;;
  *)
    echo "usage: $0 [smoke|single|concurrency4|full]" >&2
    exit 2
    ;;
esac

python "${PROJECT_DIR}/probe_vllm_attention_env.py" > "${OUT_DIR}/vllm_attention_env.json"

for CONFIG in "${CONFIGS[@]}"; do
  NAME="$(basename "${CONFIG}" .yaml)"
  (
    cd "${BENCH_DIR}"
    python benchmark.py \
      --config "${CONFIG}" \
      --output-csv "${OUT_DIR}/${NAME}.csv" \
      --output-json "${OUT_DIR}/${NAME}.json"
  ) 2>&1 | tee "${OUT_DIR}/${NAME}.log"
done

echo "${OUT_DIR}"
