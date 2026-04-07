# External Baseline Tracing

This is the canonical runbook for tracing and inspecting the local external
baseline stacks we use on RTX 5090:

- `vLLM`
- `TRT-LLM` PyTorch serve for NemotronH

Project-specific results and one-off experiments should stay in the active
`proj-*` notes. Reusable tracing instructions belong here.

## Top-Level Summary

There is no dedicated top-level `RUNBOOK.md` in this repo. The top-level
[README.md](/home/khkramer/src/nemotron-inference/README.md) is the discovery
page, and this document is the detailed tracing runbook for external
baselines.

## TRT-LLM NemotronH Trace Hooks

These hooks are for the local source build under
[third_party/TensorRT-LLM](/home/khkramer/src/nemotron-inference/third_party/TensorRT-LLM).

Important: on the local NemotronH serve path, the live routed MoE path is the
PyTorch CUTLASS fused-MoE custom op, not the older `trtllmGen` grouped-GEMM
runner.

Relevant files:

- [torch_custom_ops.py](/home/khkramer/src/nemotron-inference/third_party/TensorRT-LLM/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py)
- [moeOp.cpp](/home/khkramer/src/nemotron-inference/third_party/TensorRT-LLM/cpp/tensorrt_llm/thop/moeOp.cpp)
- [run_trtllm_nano_serve.sh](/home/khkramer/src/nemotron-inference/proj-2026-04-05-1704/run_trtllm_nano_serve.sh)
- [trtllm_nano_serve.yaml](/home/khkramer/src/nemotron-inference/proj-2026-04-05-1704/trtllm_nano_serve.yaml)

### Trace Tactic IDs

Set:

```bash
export TLLM_FUSED_MOE_PRINT_TACTICS=1
```

This prints the selected `gemm1` and `gemm2` tactic IDs for each observed MoE
shape on the live serve path.

### Trace Full CUTLASS Descriptors

Set:

```bash
export TLLM_FUSED_MOE_PRINT_TACTIC_DESCRIPTORS=1
```

This prints the selected `CutlassGemmConfig::toString()` descriptors from the
live fused-MoE path, including tile shape, cluster shape, `swap_ab`, and
epilogue-fusion information.

If `moeOp.cpp` changed, rebuild the TRT-LLM Torch op library:

```bash
cmake --build artifacts/trtllm_source_build10/cpp-build --target th_common -j 8
```

### Run The Local TRT-LLM Server With Tracing

```bash
TLLM_FUSED_MOE_PRINT_TACTICS=1 \
TLLM_FUSED_MOE_PRINT_TACTIC_DESCRIPTORS=1 \
./proj-2026-04-05-1704/run_trtllm_nano_serve.sh
```

Then drive a request through the local server. The current local benchmark and
smoke path is the OpenAI-compatible serve path.

### Current Recovered TRT-LLM Routed MoE Tactics

Artifacts:

- [trtllm_fused_moe_tactics_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trtllm_fused_moe_tactics_20260407.log)
- [trtllm_fused_moe_tactic_descriptors_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trtllm_fused_moe_tactic_descriptors_20260407.log)

Observed live selections for the local Nano path:

- `prefix4`
  - `gemm1=5`: TMA Warp Specialized, `128x128x64`, `swap_ab=true`,
    `epilogue_fusion=0`
  - `gemm2=15`: TMA Warp Specialized, `256x128x64`, `swap_ab=true`,
    `epilogue_fusion=1`
- `prefix128`
  - `gemm1=4`: TMA Warp Specialized, `128x128x128`, `swap_ab=true`,
    `epilogue_fusion=0`
  - `gemm2=13`: TMA Warp Specialized, `128x128x64`, `swap_ab=true`,
    `epilogue_fusion=1`
- `prefix4096`
  - `gemm1=1`: TMA Warp Specialized, `128x128x64`, `swap_ab=false`,
    `epilogue_fusion=0`
  - `gemm2=13`: TMA Warp Specialized, `128x128x64`, `swap_ab=true`,
    `epilogue_fusion=1`

Packed live shapes observed on the real path:

- input shape `(tokens, 1344)`
- `fc1_shape=(128, 1920, 168)`
- `fc2_shape=(128, 2688, 120)`

## vLLM Notes

The local vLLM baseline runbook is in
[docs/vllm_rtx5090.md](/home/khkramer/src/nemotron-inference/docs/vllm_rtx5090.md).

Project-specific vLLM cache probes, TTFT runs, and external baseline
comparisons should be recorded in the active project notes rather than here.
