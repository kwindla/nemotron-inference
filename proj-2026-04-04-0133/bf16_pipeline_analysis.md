# BF16 Pipeline Conversion Analysis

## Current State vs vLLM

### What matches (no change needed)
- KV cache: both BF16
- Attention/Mamba weight dtypes: both BF16/NVFP4 as stored
- GEMM accumulation: both FP32 for dense and NVFP4
- MoE gate: both FP32
- MoE routing: both FP32

### What mismatches (the actual divergence sources)

| Data path | Runtime | vLLM | Impact |
|-----------|---------|------|--------|
| Embedding output | FP32 | BF16 | First layer sees different precision |
| Hidden state buffers | FP32 | BF16 | All inter-layer data is wider |
| Residual buffers | FP32 | BF16 | Residual stream accumulates differently |
| Dense GEMM input | FP32 (promoted from BF16 weight) | BF16 natively | torch.nn.functional.linear BF16 path |
| RMSNorm input/output | FP32 in, FP32 out | BF16 in, fused add+norm, BF16 out |
| LM head input | FP32 | BF16 | Final logit computation |
| Mamba recurrent state | FP32 | configurable (FP32 for this model) | May already match |

### Key insight
The GEMM cores match (FP32 accum). The mismatch is in the **data flowing between layers**:
vLLM: BF16 → RMSNorm(FP32 internal) → BF16 → GEMM(FP32 accum) → BF16 → residual add(BF16) → repeat
Ours:  FP32 → RMSNorm(FP32)         → FP32 → GEMM(FP32)        → FP32 → residual add(FP32) → repeat

The BF16 truncation at each layer boundary is what vLLM does. Our FP32 pipeline preserves more precision per step but accumulates a *different* rounding pattern. After 52 layers, the logit distributions diverge enough to flip close argmax decisions.

## Plan Outline

### 8a. BF16 hidden/residual buffers and embedding output
- Change `RequestExecutionContext` hidden/residual/scratch from `DeviceTensorFp32` to `DeviceTensorBf16`
- Change embedding table lookup to output BF16 directly (stop FP32 promotion)
- Update `RunTokens()` in `single_token_forward_model.cpp` to pass BF16 tensors
- This is the foundation — everything else depends on BF16 data flowing between layers

### 8b. Fused add+RMSNorm kernel
- Implement a kernel matching vLLM's `fused_add_rms_norm`: takes BF16 hidden + BF16 residual, does FP32 add + FP32 variance + FP32 normalize + FP32 weight multiply, outputs BF16 norm result and updated BF16 residual
- Replace the separate `RmsNorm` + `ResidualAddFp32` pattern in attention_layer.cpp, mamba_layer.cpp, expert_layer.cpp
- This matches vLLM's precision model: internal FP32 math, BF16 boundary values

### 8c. BF16 dense linear path
- Dense GEMM (attention Q/K/V/O, embedding projection, LM head): accept BF16 input, BF16 weights, FP32 accumulation, BF16 output
- Stop converting BF16 weights to FP32 in `DeviceDenseWeightFp32::Upload()`; keep a `DeviceDenseWeightBf16` path
- cuBLASLt supports BF16 input/output with FP32 compute: `CUBLAS_COMPUTE_32F` with `CUDA_R_16BF` data types

### 8d. NVFP4 linear BF16 I/O
- NVFP4 GEMM already uses FP32 accumulation — ensure input activation quantization starts from BF16 (not FP32)
- Ensure output is cast to BF16 before writing to hidden buffer
- This should be a small change since the GEMM core already matches

### 8e. Layer interface updates
- Update `AttentionLayerSlice::Run()`, `MambaLayerSlice::Run()`, `ExpertLayerSlice::Run()` signatures to accept BF16 input/output
- Update the residual-add pattern to match vLLM's deferred fused pattern
- Verify Mamba recurrent state stays FP32 (vLLM's `mamba_ssm_cache_dtype` defaults to FP32 for this model)

### 8f. Bootstrap/lifecycle cleanup
- Split `BuildFromManifestFile()` into validation/planning/assembly
- Replace heuristic layer-role inference with explicit manifest roles
- Formalize request persistent state vs execution scratch

## SM120 Considerations

Need to verify for each sub-step:
- cuBLASLt BF16 matmul support on SM120 (should be fine — BF16 has been supported since SM80)
- Fused add+norm kernel: need to check if we can use vLLM's kernel directly or need our own
- NVFP4 activation quantization from BF16 vs FP32: check if cuBLASLt NVFP4 expects FP32 or can take BF16

## Adversarial Review Findings

### Blockers identified
1. **Layer Run() signatures hardcoded to DeviceTensorFp32** — AttentionLayerSlice, MambaLayerSlice, ExpertLayerSlice all take `const DeviceTensorFp32&`. Not templated. Every layer must change.
2. **NVFP4 activation packing is FP32-only** — `PackRowMajorFp32ToNvfp4()` takes `const float*`. BF16 activations need explicit BF16→FP32 cast before packing. vLLM does the same thing implicitly inside `scaled_fp4_quant`.
3. **Embedding table converts BF16→FP32 on upload** — no BF16 lookup kernel exists. Need new kernel.
4. **No fused add+RMSNorm kernel** — current RmsNormKernel is FP32-only, separate from residual add.

### What already works
- `DeviceTensorBf16` class exists with full Create/CopyFrom/CopyTo API
- KV cache already BF16
- Mamba recurrent state can stay FP32 (confirmed by vLLM config)
- cuBLASLt supports BF16 I/O + FP32 compute (CUDA_R_16BF + CUBLAS_COMPUTE_32F)

### Revised approach
Instead of templating all layers, use **BF16 as the inter-layer dtype with FP32 internal compute**:
- Hidden/residual buffers: BF16 (matches vLLM)
- Layer interfaces: change to `const DeviceTensorBf16&` input/output
- Inside each layer: cast BF16→FP32 where needed for GEMM input prep
- Dense GEMM: cuBLASLt with CUDA_R_16BF data + CUBLAS_COMPUTE_32F compute (single enum change)
- NVFP4 GEMM: BF16 → cast to FP32 → pack to FP4 → GEMM → cast output to BF16
- RMSNorm: fused kernel (BF16 in, FP32 internal, BF16 out)
- Mamba state: stays FP32, cast at boundaries

## Revised Plan Outline (sub-steps of step 8)

### 8a. Fused add+RMSNorm kernel (BF16 I/O, FP32 internal)
Foundation kernel needed by all layers. Write once, use everywhere.
- Input: BF16 hidden, BF16 residual, FP32 norm weight, float epsilon
- Internal: FP32 add, FP32 variance, FP32 normalize, FP32 weight multiply
- Output: BF16 normalized result, BF16 updated residual (in-place)
- Test: compare against separate FP32 RmsNorm + ResidualAdd with tolerance

### 8b. BF16 embedding lookup
- Keep BF16 weights on device (stop FP32 promotion during upload)
- New `LookupEmbeddingRowsBf16()` that outputs BF16 directly
- Small kernel change: read BF16, write BF16

### 8c. BF16 dense GEMM path
- cuBLASLt: change data type enums from CUDA_R_32F to CUDA_R_16BF
- Keep CUBLAS_COMPUTE_32F for FP32 accumulation
- Add `DeviceDenseWeightBf16` that keeps BF16 on device (parallel to existing FP32 path)
- Test: compare BF16 GEMM output against FP32 GEMM with tolerance

### 8d. Layer interface conversion (BF16 I/O)
- Change Run() signatures in attention_layer.h, mamba_layer.h, expert_layer.h
  from DeviceTensorFp32 to DeviceTensorBf16
- Change RequestExecutionContext hidden/residual/scratch from FP32 to BF16
- Inside each layer: use fused add+RMSNorm from 8a, BF16 GEMMs from 8c
- Mamba: cast BF16 activations to FP32 for recurrent state ops, cast back
- NVFP4 expert path: cast BF16 input to FP32 for activation packing, keep rest

### 8e. Residual-add pattern alignment
- Remove eager residual add from attention_layer.cpp, mamba_layer.cpp, expert_layer.cpp
- Each layer returns its output in BF16; the caller (RunTokens) manages the residual stream
- Fused add+RMSNorm at the start of each layer handles add + norm in one kernel

### 8f. End-to-end verification
- Regenerate oracle, run vLLM parity — target 16/16 match
- Run full ctest, verify 53/53 pass
- Run decode benchmark, verify no regression (expect improvement from BF16 bandwidth)

## SM120 Considerations (verified)
- cuBLASLt BF16 matmul: supported on all SM >= 80, including SM120 ✓
- CUDA_R_16BF data type: available in CUDA 13.0 ✓  
- __nv_bfloat16 type: available since CUDA 11.0 ✓
- Fused RMSNorm: standard CUDA kernel, no SM-specific concerns ✓
- NVFP4 activation packing: FP32 input required, BF16→FP32 cast is cheap ✓
