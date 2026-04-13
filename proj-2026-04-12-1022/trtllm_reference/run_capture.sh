#!/usr/bin/env bash
# Step 2b helper: apply the flashinfer BF16 dump patch, run the capture harness,
# and revert the patch (always, via EXIT trap).
#
# Auto-detects the flashinfer source tree that the warm JIT cache is wired to,
# patches THAT tree, and runs from THAT venv so the cached ninja build picks up
# the patch. Fails loudly if detection is ambiguous.
#
# Usage:
#   bash proj-2026-04-12-1022/trtllm_reference/run_capture.sh
#
# Problem-shape knobs (env vars):
#   NEMOTRON_HARNESS_M        num_tokens          (default 128)
#   NEMOTRON_HARNESS_K        hidden_size         (default 256)
#   NEMOTRON_HARNESS_N        inter_size          (default 256)
#   NEMOTRON_HARNESS_E        num_experts         (default 1)
#   NEMOTRON_HARNESS_TOPK     top_k               (default 1)
#   NEMOTRON_HARNESS_SEED     torch.manual_seed   (default 0xC0FFEE)
set -euo pipefail

REPO_ROOT="/home/khkramer/src/nemotron-inference"
PROJ_REF_DIR="${REPO_ROOT}/proj-2026-04-12-1022/trtllm_reference"
PATCH_FILE="${PROJ_REF_DIR}/patches/flashinfer_bf16_gemm1_dump.patch"
BUILD_NINJA="${HOME}/.cache/flashinfer/0.6.6/120a/cached_ops/fused_moe_120/build.ninja"
GOLDEN_DIR="${PROJ_REF_DIR}/golden"
METADATA_PATH="${GOLDEN_DIR}/bf16_gemm1_metadata.json"
INPUT_SAVE_DIR="${GOLDEN_DIR}"

M="${NEMOTRON_HARNESS_M:-128}"
K="${NEMOTRON_HARNESS_K:-256}"
N="${NEMOTRON_HARNESS_N:-256}"
E="${NEMOTRON_HARNESS_E:-1}"
TOPK="${NEMOTRON_HARNESS_TOPK:-1}"
SEED="${NEMOTRON_HARNESS_SEED:-12648430}"

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "ERROR: patch file not found at ${PATCH_FILE}" >&2
  exit 1
fi

# --- Auto-detect the flashinfer source tree that the warm JIT cache is wired to.
# The warm ninja build file bakes absolute source paths into every `cuda_compile`
# line. Parse one of those lines and extract the venv root.
if [[ ! -f "${BUILD_NINJA}" ]]; then
  echo "ERROR: no warm flashinfer fused_moe_120 JIT cache at ${BUILD_NINJA}" >&2
  echo "       Run flashinfer.fused_moe.cutlass_fused_moe once from the venv" >&2
  echo "       you want to patch so the cache is generated, then re-run this script." >&2
  exit 1
fi

# Extract the first venv-rooted source path the ninja build references.
FLASHINFER_SRC_ROOT="$(
  awk '
    /cuda_compile .*\/flashinfer\/data\/csrc\// {
      for (i = 1; i <= NF; ++i) {
        if ($i ~ /\/flashinfer\/data\/csrc\//) {
          sub(/\/flashinfer\/data\/csrc\/.*$/, "/flashinfer/data/csrc", $i)
          print $i
          exit
        }
      }
    }
  ' "${BUILD_NINJA}"
)"

if [[ -z "${FLASHINFER_SRC_ROOT}" || ! -d "${FLASHINFER_SRC_ROOT}" ]]; then
  echo "ERROR: could not extract flashinfer src root from ${BUILD_NINJA}" >&2
  exit 1
fi
echo "[run_capture] warm JIT cache points at flashinfer src root:"
echo "[run_capture]   ${FLASHINFER_SRC_ROOT}"

# Confirm every csrc build line in the ninja file agrees on this same root.
MISMATCH_COUNT="$(
  awk -v root="${FLASHINFER_SRC_ROOT}" '
    /cuda_compile .*\/flashinfer\/data\/csrc\// {
      for (i = 1; i <= NF; ++i) {
        if ($i ~ /\/flashinfer\/data\/csrc\//) {
          if (index($i, root) != 1) { mismatch++; break }
        }
      }
    }
    END { print mismatch + 0 }
  ' "${BUILD_NINJA}"
)"
if [[ "${MISMATCH_COUNT}" != "0" ]]; then
  echo "ERROR: ${MISMATCH_COUNT} build lines in the warm ninja file reference" >&2
  echo "       different flashinfer source trees. Refusing to patch an ambiguous tree." >&2
  echo "       Delete ${HOME}/.cache/flashinfer/0.6.6/120a/cached_ops/fused_moe_120/" >&2
  echo "       and re-run this script to force a clean build from a single source tree." >&2
  exit 1
fi

# Derive the venv root (strip /lib/python*/site-packages/flashinfer/data/csrc).
VENV_ROOT="$(echo "${FLASHINFER_SRC_ROOT}" | sed -E 's#/lib/python[^/]+/site-packages/flashinfer/data/csrc##')"
VENV_PYTHON="${VENV_ROOT}/bin/python"
PATCH_TARGET="${FLASHINFER_SRC_ROOT}/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: resolved venv python does not exist at ${VENV_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${PATCH_TARGET}" ]]; then
  echo "ERROR: patch target not found at ${PATCH_TARGET}" >&2
  exit 1
fi
echo "[run_capture] venv python:  ${VENV_PYTHON}"
echo "[run_capture] patch target: ${PATCH_TARGET}"

# --- Clean stale artifacts BEFORE the run so stale dumps cannot false-pass.
mkdir -p "${GOLDEN_DIR}"
rm -f \
  "${GOLDEN_DIR}"/bf16_gemm1_tactic*.bin \
  "${GOLDEN_DIR}"/bf16_gemm1_metadata.json \
  "${GOLDEN_DIR}"/inputs.pt \
  "${GOLDEN_DIR}"/final_moe_output.pt
echo "[run_capture] cleared stale artifacts under ${GOLDEN_DIR}"

revert_patch() {
  echo "[run_capture] reverting flashinfer patch..."
  (cd "${FLASHINFER_SRC_ROOT}" && patch -p1 -R --silent < "${PATCH_FILE}") || {
    echo "[run_capture] WARNING: patch revert failed; inspect ${PATCH_TARGET}" >&2
  }
}
trap revert_patch EXIT

echo "[run_capture] applying flashinfer BF16 dump patch..."
(cd "${FLASHINFER_SRC_ROOT}" && patch -p1 < "${PATCH_FILE}")

echo "[run_capture] touching patch target so ninja invalidates the affected .o files..."
touch "${PATCH_TARGET}"

echo "[run_capture] running harness:"
echo "  M=${M} K=${K} N=${N} E=${E} topk=${TOPK} seed=${SEED}"
stdbuf -oL -eL "${VENV_PYTHON}" "${PROJ_REF_DIR}/capture_bf16_gemm1.py" \
  --num-tokens "${M}" \
  --hidden-size "${K}" \
  --inter-size "${N}" \
  --num-experts "${E}" \
  --top-k "${TOPK}" \
  --seed "${SEED}" \
  --golden-dir "${GOLDEN_DIR}" \
  --metadata-path "${METADATA_PATH}" \
  --input-save-dir "${INPUT_SAVE_DIR}" \
  --flashinfer-src-root "${FLASHINFER_SRC_ROOT}" \
  --venv-root "${VENV_ROOT}"

echo "[run_capture] capture complete. Dump files:"
ls -la "${GOLDEN_DIR}"/bf16_gemm1_tactic*.bin 2>&1 || echo "  (no tactic dumps found!)"
ls -la "${METADATA_PATH}" "${INPUT_SAVE_DIR}/inputs.pt" 2>&1 || true
