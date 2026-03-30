#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${ROOT}/artifacts/preflight"
MODEL_DIR="${MODEL_DIR:-/home/khkramer/models/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1}"
HF_REPO_ID="${HF_REPO_ID:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
VLLM_IMAGE="${VLLM_IMAGE:-nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065}"
RUN_VLLM_SMOKE="${RUN_VLLM_SMOKE:-0}"

mkdir -p "${ARTIFACT_DIR}"

python3 - <<'PY' "${ARTIFACT_DIR}/environment_report.json" "${VLLM_IMAGE}"
import json
import subprocess
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
vllm_image = sys.argv[2]

def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

report = {
    "python": run(["python3", "--version"]),
    "cmake": run(["cmake", "--version"]),
    "nvcc": run(["nvcc", "--version"]),
    "nvidia_smi": run(["nvidia-smi", "-L"]),
    "docker": run(["docker", "--version"]),
    "cuda_headers": {
        "cublasLt": Path("/usr/local/cuda/include/cublasLt.h").exists(),
        "cudnn": Path("/usr/local/cuda/include/cudnn.h").exists(),
    },
    "docker_image": run(["docker", "image", "inspect", vllm_image]),
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cmake -S "${ROOT}" -B "${ROOT}/build" >/dev/null
cmake --build "${ROOT}/build" --target cublaslt_probe -j >/dev/null
set +e
"${ROOT}/build/tools/probes/cublaslt_probe" \
  > "${ARTIFACT_DIR}/cublaslt_probe.json" \
  2> "${ARTIFACT_DIR}/cublaslt_probe.stderr.txt"
probe_rc=$?
set -e
printf '%s\n' "${probe_rc}" > "${ARTIFACT_DIR}/cublaslt_probe.exitcode"

docker run --rm \
  --entrypoint python \
  -v "${ROOT}:/workspace" \
  -v "${MODEL_DIR}:/model:ro" \
  -w /workspace \
  "${VLLM_IMAGE}" \
  tools/inspect_checkpoint/inspect_checkpoint.py \
    --repo-id "${HF_REPO_ID}" \
    --model-dir /model \
    --output artifacts/preflight/checkpoint_report.json

python3 "${ROOT}/tools/memory_budget/calc_memory_budget.py" \
  --checkpoint-report "${ARTIFACT_DIR}/checkpoint_report.json" \
  --output-json "${ARTIFACT_DIR}/memory_budget_report.json" \
  --output-md "${ARTIFACT_DIR}/memory_budget_report.md"

docker run --rm \
  --entrypoint python \
  -v "${ROOT}:/workspace" \
  -v "${MODEL_DIR}:/model:ro" \
  -w /workspace \
  "${VLLM_IMAGE}" \
  tools/oracle/tokenize_prompts.py \
    --model-dir /model \
    --prompts testing/oracle/prompts.json \
    --output artifacts/preflight/tokenization_vectors.json

if [[ "${RUN_VLLM_SMOKE}" == "1" ]]; then
  "${ROOT}/benchmarks/ttft_bench/smoke_vllm_baseline.sh"
fi

printf 'Preflight artifacts written to %s\n' "${ARTIFACT_DIR}"
