#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Dual-serve: Nemotron 3 Super (NVFP4) + Nemotron 3 Nano (FP8)
#
# GPU memory budget on GB10 (121.7 GiB usable unified):
#   Super NVFP4  ~75 GB weights  → gpu-memory-utilization 0.64 (~78 GB)
#   Nano  FP8    ~30 GB weights  → gpu-memory-utilization 0.25 (~30 GB)
#   Total: 0.89 → ~108 GB, leaving ~13 GB for system/display
#   Note: MAX_JOBS=2 limits FlashInfer JIT parallelism to avoid OOM during
#   CUDA kernel compilation (~40 min first start, cached thereafter).
#
# Models are started sequentially so the first container's CUDA allocation
# is visible to the second.  Expect ~20 min for Super init, ~5 min for Nano.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Containers ---
SUPER_CONTAINER="nemotron3-super-nvfp4"
NANO_CONTAINER="nemotron3-nano-fp8"

# --- Images ---
# The -bf16-fp8 EA image supports MIXED_PRECISION quant (FP8+NVFP4) and has
# the Super model baked in.  The base :0.1.0 image does NOT support it.
SUPER_IMAGE="nvcr.io/0767305323357365/ea-nemotron-3-super/nemotron-3-super:0.1.0-bf16-fp8"
NANO_IMAGE="nvcr.io/0767305323357365/ea-nemotron-3-super/nemotron-3-super:0.1.0-bf16-fp8"

# --- Models ---
# Super: mounted from host /home/khkramer/models → /model inside container
SUPER_MODEL_PATH="/model/ea_final_nvidia_nemotron_3_super_120b_a12b_nvfp475_030326_vv0.1"
NANO_MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"   # resolved from HF cache
MODELS_DIR="/home/khkramer/models"

# --- Ports ---
SUPER_PORT=8000
NANO_PORT=8002
ROUTER_PORT=8080

# --- GPU memory split (must sum ≤ 0.95) ---
SUPER_GPU_MEM=0.64
NANO_GPU_MEM=0.30

# --- Shared volumes (matching the working bf16-fp8 container) ---
HF_CACHE="/home/khkramer/.cache/huggingface"
VLLM_CACHE="/home/khkramer/.cache/vllm_ubuntu"
FLASHINFER_CACHE="/home/khkramer/.cache/flashinfer"

# ---- helpers ---------------------------------------------------------------

wait_for_ready() {
    local name="$1" url="$2" timeout="${3:-1800}"
    local elapsed=0
    echo "  Waiting for ${name} to be ready (up to $((timeout/60)) min)..."
    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "${url}/health" >/dev/null 2>&1; then
            echo "  ${name} is ready."
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        # Print progress every 60s
        if (( elapsed % 60 == 0 )); then
            echo "  ... ${elapsed}s elapsed"
        fi
    done
    echo "  WARNING: ${name} not ready after ${timeout}s — continuing anyway."
    return 1
}

# ---- pre-flight checks -----------------------------------------------------

# The existing bf16-fp8 container uses 90% of GPU memory and must be stopped.
if docker ps --format '{{.Names}}' | grep -q 'nemotron3-super-ea-bf16-fp8-vllm-uncached'; then
    echo "Stopping nemotron3-super-ea-bf16-fp8-vllm-uncached (frees GPU memory)..."
    docker stop nemotron3-super-ea-bf16-fp8-vllm-uncached >/dev/null
fi

docker rm -f "${SUPER_CONTAINER}" >/dev/null 2>&1 || true
docker rm -f "${NANO_CONTAINER}" >/dev/null 2>&1 || true

mkdir -p "${HF_CACHE}" "${VLLM_CACHE}" "${FLASHINFER_CACHE}" /home/khkramer/logs

# ---- 1. Launch Super NVFP4 on port $SUPER_PORT ----------------------------

echo ""
echo "=== Starting Nemotron 3 Super NVFP4 on port ${SUPER_PORT} ==="
docker run -d \
    --name "${SUPER_CONTAINER}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "${SUPER_PORT}:8000" \
    -e VLLM_FLASHINFER_MOE_BACKEND=throughput \
    -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
    -e VLLM_USE_FLASHINFER_MOE_FP8=1 \
    -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
    -e MAX_JOBS=2 \
    -e HOME=/home/ubuntu \
    -e HF_HOME=/home/ubuntu/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/home/ubuntu/.cache/huggingface/hub \
    -e HF_MODULES_CACHE=/home/ubuntu/.cache/huggingface/modules \
    -e TRANSFORMERS_CACHE=/home/ubuntu/.cache/huggingface/transformers \
    -v "${MODELS_DIR}:/model" \
    -v "${HF_CACHE}:/home/ubuntu/.cache/huggingface" \
    -v "${VLLM_CACHE}:/home/ubuntu/.cache/vllm" \
    -v "${FLASHINFER_CACHE}:/home/ubuntu/.cache/flashinfer" \
    --entrypoint /usr/local/bin/vllm \
    "${SUPER_IMAGE}" \
    serve "${SUPER_MODEL_PATH}" \
    --served-model-name nemotron-3-super \
    --async-scheduling \
    --tensor-parallel-size 1 \
    --swap-space 0 \
    --trust-remote-code \
    --gpu-memory-utilization "${SUPER_GPU_MEM}" \
    --enable-expert-parallel \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --attention-backend TRITON_ATTN \
    --enforce-eager \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

wait_for_ready "Super NVFP4" "http://localhost:${SUPER_PORT}" 1800

# ---- 2. Launch Nano FP8 on port $NANO_PORT --------------------------------

echo ""
echo "=== Starting Nemotron 3 Nano FP8 on port ${NANO_PORT} ==="
docker run -d \
    --name "${NANO_CONTAINER}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "${NANO_PORT}:8000" \
    -e MAX_JOBS=2 \
    -e HOME=/home/ubuntu \
    -e HF_HOME=/home/ubuntu/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/home/ubuntu/.cache/huggingface/hub \
    -e HF_MODULES_CACHE=/home/ubuntu/.cache/huggingface/modules \
    -e TRANSFORMERS_CACHE=/home/ubuntu/.cache/huggingface/transformers \
    -v "${HF_CACHE}:/home/ubuntu/.cache/huggingface" \
    -v "${VLLM_CACHE}:/home/ubuntu/.cache/vllm" \
    -v "${FLASHINFER_CACHE}:/home/ubuntu/.cache/flashinfer" \
    --entrypoint /usr/local/bin/vllm \
    "${NANO_IMAGE}" \
    serve "${NANO_MODEL_ID}" \
    --served-model-name nemotron-3-nano \
    --async-scheduling \
    --tensor-parallel-size 1 \
    --swap-space 0 \
    --trust-remote-code \
    --gpu-memory-utilization "${NANO_GPU_MEM}" \
    --enable-expert-parallel \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --attention-backend TRITON_ATTN \
    --enforce-eager \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

wait_for_ready "Nano FP8" "http://localhost:${NANO_PORT}" 2400

# ---- 3. Start router -------------------------------------------------------

echo ""
echo "============================================="
echo "  Nemotron Dual-Serve Ready"
echo "============================================="
echo "  Super NVFP4 : http://localhost:${SUPER_PORT}/v1  (model: nemotron-3-super)"
echo "  Nano  FP8   : http://localhost:${NANO_PORT}/v1  (model: nemotron-3-nano)"
echo "  Router      : http://localhost:${ROUTER_PORT}/v1  (auto-routes by model name)"
echo "============================================="
echo ""
echo "Starting router (Ctrl-C to stop)..."
exec python3 "${SCRIPT_DIR}/router.py" "${ROUTER_PORT}"
