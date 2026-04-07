# GB10 Performance Planning

## Purpose

This document is the performance-planning sub-project for the Nemotron runtime on DGX Spark / GB10.

Execution history for this sub-project is tracked separately in [gb10_performance_progress.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/docs/gb10_performance_progress.md).

Its job is to answer one question before we lock more runtime data structures:

- are we making the right low-level memory-layout, numeric-format, and operator-selection decisions for GB10 given the actual service priorities?

Current service priorities:

1. low TTFT
2. decode speed

This document is intentionally separate from the main implementation plan because these decisions affect:

- in-memory weight layout
- request-state layout
- reusable-cache layout
- numeric-format policy
- which library backends we should trust on GB10

## Current Position

We have enough evidence to justify the current runtime direction, but not enough evidence to freeze all GB10-specific layout decisions.

What is already justified:

- move repeated hot weights to device-resident startup-owned storage
- preserve per-tensor dtype and layout metadata
- design KV state around paged attention
- treat Mamba recurrent-state bandwidth as a first-class decode optimization problem
- treat `cuBLASLt` and `cuDNN FE` as hard GB10 constraints, not soft preferences

What is not yet justified:

- freezing the final NVFP4 packed-weight layout details for GB10 beyond the currently validated row-major `row_nt_mk_nk` contract
- freezing the final dense-weight residency policy for all operator families
- freezing the final Mamba-state cache representation for decode
- making claims about TTFT or decode-optimal operator choices without GB10 shape-family measurements

## Known Constraints

### DGX Spark Memory System

DGX Spark provides:

- `128 GB` LPDDR5x unified system memory
- `273 GB/s` memory bandwidth

This is a unified-memory platform, not a high-bandwidth datacenter HBM environment.

Implication:

- avoid repeated hot-path weight copies
- minimize request-local and cache-state bandwidth
- assume decode will become memory-bound quickly
- use two bandwidth figures in planning:
  - `273 GB/s` theoretical peak
  - about `180 GB/s` as a more realistic decode-planning number under inference load
- treat colocated CPU work as a major bandwidth risk during measurement and serving

Sources:

- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf)
- [DGX Spark Hardware Guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)

### SM121 Software Envelope

GB10 is `SM121`, not datacenter `SM100`.

Implication:

- no TMEM
- no `tcgen05`
- no WGMMA
- smaller shared-memory and cache envelope than datacenter Blackwell
- `cuBLASLt` is the required dense-GEMM foundation on this target
- `cuDNN FE` is the required attention foundation on this target
- CUTLASS SM100 examples and FlashAttention-family backends should be treated as non-portable or broken on SM121 unless proven otherwise on GB10 specifically

This is a hard architectural constraint, not just a preference ordering.

### UMA / Porting Constraints

NVIDIA's porting material for DGX Spark explicitly calls out unified-memory behavior and platform-specific caveats.

Important implications:

- budgeting and residency assumptions must be GB10-aware
- `cudaMemGetInfo` is not sufficient as the only memory-budget signal on this platform
- GPUDirect RDMA is not available, so we should not design around that path
- GPU OOM must be treated as an operational safety issue, not just a recoverable error path:
  - under unified-memory pressure DGX Spark can become unresponsive instead of returning a clean CUDA OOM
  - the planner should use `/proc/meminfo`-based signals and preserve a documented safety margin
  - benchmark hygiene may require cache dropping or swap reclamation outside the runtime, but those are operator workflows, not runtime dependencies

Sources:

- [DGX Spark Porting Guide: Optimization](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html)
- [DGX Spark Porting Guide: CUDA](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
- [DGX Spark Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)

### CUDA Toolchain Constraint

The local measurement stack is now rebuilt against CUDA `13.2` and uses the CUDA compat user-mode driver from `/usr/local/cuda-13.2/compat` when tests and benchmark wrappers run. GB10 performance and correctness work should use that path, not the earlier CUDA `13.0` linkage.

Reasons:

- `cuBLASLt` BF16/FP16 illegal-memory-access bugs on DGX Spark were fixed in `13.2`
- `13.2` also carries DGX Spark-specific `NVFP4` / `MXFP8` improvements
- older CUDA `13.0` benchmark artifacts should now be treated as provisional historical baselines only

## Model-Specific Constraints

### Mixed Precision Is Intentional

The official checkpoint is mixed precision. It is not correct to design the runtime as if all forward-pass weights should be normalized into one universal low-precision format.

The model card and report indicate a mixed deployment recipe across:

- NVFP4
- MXFP8 / FP8
- BF16
- FP32 for selected state / scale behavior

Implication:

- the runtime must preserve per-tensor format identity through packing, loading, and execution planning
- we should not "simplify" the runtime by flattening everything to FP32 or everything to NVFP4

Sources:

- [Nemotron 3 Super NVFP4 Model Card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
- [Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

### Mamba State Bandwidth Matters For Decode

The technical report is explicit that Mamba state-cache DRAM reads become a decode bottleneck, and that naive FP16 casting harmed verbosity. NVIDIA's selected direction was FP16 cache storage with stochastic rounding during conversion.

Implication:

- Mamba-state layout and type conversion are part of decode optimization, not a secondary memory-saving detail
- cache representation must be benchmarked as a decode-speed decision

Source:

- [Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

## Current Local Evidence

### Preflight Findings

Local preflight already showed:

- host GPU detected as `NVIDIA GB10`
- initial local host CUDA compiler reported `13.0`
- local `cuBLASLt` headers were present
- local cuDNN development headers were missing during the first preflight
- the first naive `cuBLASLt` FP4 heuristic probe failed with `CUBLAS_STATUS_NOT_SUPPORTED`

Current local artifacts:

- [preflight_summary.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/preflight/preflight_summary.md)
- [cublaslt_probe.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/preflight/cublaslt_probe.json)
- [gb10_dense_gemm_default_local_cuda13000.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.json)
- [gb10_dense_gemm_default_local_cuda13000.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.env.txt)
- [gb10_nvfp4_contract_default_local_cuda13000.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_contract_default_local_cuda13000.json)
- [gb10_nvfp4_contract_default_local_cuda13000.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_contract_default_local_cuda13000.env.txt)

Implication:

- the first guessed FP4 descriptor/layout contract was not sufficient
- a later local contract matrix did find repeatable successful paths on CUDA `13.0`:
  - `row_nt_mk_nk`
  - `col_tn_km_kn`
- the runtime-relevant row-major path is now strong enough to encode explicitly in the planner and upload surfaces
- the local toolchain has since moved past that preflight state and now exposes:
  - CUDA `13.2`
  - cuDNN `9.20.0`
  - vendored cuDNN Frontend headers
  - a first executable BF16 cuDNN FE paged-attention runtime path
- the earlier `13.0` result was useful for narrowing the contract search, but it was not strong enough to freeze the runtime contract on its own

### CUDA 13.2 Rebuild And Measurement Rerun

The local runtime and benchmark binaries are now rebuilt against `/usr/local/cuda-13.2`, and the test and benchmark surfaces prepend the compat directory automatically when it exists.

Current CUDA `13.2` artifacts:

- [gb10_dense_gemm_default_20260328T222445Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_dense_gemm_default_20260328T222445Z_cuda132.json)
- [gb10_dense_gemm_default_20260328T222445Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_dense_gemm_default_20260328T222445Z_cuda132.env.txt)
- [gb10_nvfp4_contract_default_20260328T222445Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_contract_default_20260328T222445Z_cuda132.json)
- [gb10_nvfp4_contract_default_20260328T222445Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_contract_default_20260328T222445Z_cuda132.env.txt)
- [gb10_loader_modes_20260328T222550Z.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_loader_modes_20260328T222550Z.json)
- [gb10_loader_modes_20260328T222550Z.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_loader_modes_20260328T222550Z.env.txt)

Measured facts from the CUDA `13.2` rerun:

- dense and NVFP4 benchmark binaries now report `cuda_runtime_version=13020` and `cuda_driver_version=13020`
- the dense suite still shows strong workspace sensitivity for several TTFT-relevant families:
  - `attention_core_proj`, `m=256`: `0.896 ms` at `0` workspace vs `0.600 ms` at `4 MiB`
  - `attention_kv_proj`, `m=256`: `0.170 ms` at `0` workspace vs `0.051 ms` at `4 MiB`
  - `shared_expert_down`, `m=256`: `1.194 ms` at `0` workspace vs `0.875 ms` at `4 MiB`
  - `mamba_in_proj`, `m=256`: `6.730 ms` at `0` workspace vs `3.572 ms` at `4 MiB`
- the NVFP4 contract matrix still converges to the same result on CUDA `13.2`:
  - `row_nt_mk_nk` succeeds
  - `col_tn_km_kn` succeeds
  - naive `row_nn_mk_kn` still returns `CUBLAS_STATUS_NOT_SUPPORTED`
- the loader benchmark wrapper has now been rerun successfully with the synthetic manifest smoke fixture under the same compat-path logic

Implication:

- the runtime-facing row-major NVFP4 contract is now confirmed on the intended CUDA `13.2` user-space stack
- workspace reservation is still a first-class GB10 performance decision
- the next unknown is no longer "does the validated contract survive CUDA `13.2`?"
- the next unknown is "what does the real NVFP4 execution path do on top of that contract?"

### Current Runtime State

The runtime already has:

- device-resident FP32 embedding upload
- device-resident FP32 dense-weight upload for the first dense GEMM path
- device-side embedding lookup
- a real dense `cublasLtMatmul` path
- a composed `token ids -> embeddings -> dense projection` fragment that can run with uploaded dense weights
- a standalone dense microbenchmark harness with checkpoint-derived shape families and JSON result capture
- a standalone runtime-side NVFP4 GEMM benchmark harness with matching case names for direct comparison against the FP32 dense baseline
- a standalone NVFP4 contract-validation harness with JSON artifact capture
- a standalone loader-mode benchmark harness for `mmap` versus eager read
- explicit `cublasLt` plan encoding for the validated row-major `N/T` contract
- a startup-uploadable NVFP4 packed-weight object that preserves block scales and tensor scale as separate device buffers
- a first real runtime-side NVFP4 GEMM executor for that row-major contract, currently treating:
  - `block_scales` as the `cuBLASLt` per-block scale pointer payload
  - FP32 `tensor_scale` as an `alpha` multiplier folded in at launch time
- optional `/proc/meminfo`-based host-memory clamping in bootstrap memory budgeting
- test and benchmark surfaces that self-apply the CUDA compat path when present, so rebuilt CUDA `13.2` binaries run correctly under `ctest` and wrapper scripts

Implication:

- the first TTFT-critical hot-weight path now has a realistic startup-upload option
- benchmark work can now measure dense execution without forcing per-call weight copies
- the core dense and NVFP4 suites have now been rerun on the pinned CUDA `13.2` user-space stack
- the runtime no longer stops at NVFP4 contract validation; it now has an executable row-major FP4 path, even though that path still uses synthetic FP4 activations rather than a real runtime activation-packing pipeline
- the runtime now also has a direct benchmark surface for that executor, so NVFP4 timing no longer depends only on the standalone contract probe
- the loader benchmark path now has a wrapper-verified synthetic-manifest rerun on that same stack, but it still needs a real packed model manifest
- wider dense residency policy is still unsettled for other operator families

## Library Direction

### Dense GEMMs

The right foundation is still `cuBLASLt`, not custom dense kernels.

Reasons:

- NVIDIA documents `cuBLASLt` as the dense GEMM path
- CUDA release notes call out DGX Spark `NVFP4` / `MXFP8` GEMM improvements on newer toolchains
- our runtime already has the correct planning seam around `cuBLASLt`
- on `SM121`, other seemingly attractive Blackwell paths are either unavailable or not yet trustworthy

But:

- we should not assume all NVFP4 layout details are frozen just because one contract now works locally
- alignment, leading-dimension, scale-buffer layout, and real mixed-precision execution still need validation against the pinned `13.2+` stack and NVIDIA samples

Sources:

- [cuBLAS / cuBLASLt Documentation](https://docs.nvidia.com/cuda/archive/13.0.0/pdf/CUBLAS_Library.pdf)
- [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/pdf/CUDA_Toolkit_Release_Notes.pdf)

### Attention

`cuDNN Frontend` is not just the recommended direction on GB10; it is the only credible v1 attention backend.

It already exposes the right paged-cache model for attention on the serving path:

- paged K/V containers
- INT32 page tables
- sequence-length tensors
- decode and prefill samples

Implication:

- KV cache layout should be designed around paged attention
- do not invent a custom KV layout first and hope it maps cleanly later
- do not plan around FlashAttention / FlashInfer / FlashMLA on SM121 unless a future GB10-specific proof exists

Source:

- [cuDNN Frontend Attention](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html)

### MoE Expert Path

For GB10, the MoE priority order should be:

1. `cuDNN FE` grouped matmul / runtime fusion
2. `cuBLASLt` grouped GEMM where it fits
3. CUTLASS only as a later, GB10-proven path

Implication:

- deprioritize CUTLASS SM100 examples for `SM121`
- treat `cuBLASLt` grouped GEMM as a legitimate fallback or secondary option instead of dismissing it too early

Source:

- [cuDNN Frontend MoE Grouped Matmul](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/MoeGroupedMatmul.html)

## Numeric-Format Planning

### Weight Formats

Near-term planning assumptions:

- preserve checkpoint format identity
- use BF16 / FP32 for correctness-first paths where needed
- move toward native runtime execution in the model's intended deployment formats instead of normalizing all operators to FP32

Do not assume:

- all dense operators should run FP32 in the real runtime
- all weights should be expanded to FP32 in device memory

### Attention KV Formats

On GB10, the paged-attention target should assume `BF16` or `FP16` KV storage, not `FP8`.

Implication:

- the Nemotron deployment recipe for `FP8` KV cache should not be treated as directly portable to `SM121`
- the memory planner should derive KV bytes/token from the exact attention config and chosen GB10 dtype rather than reusing datacenter assumptions
- any older `4096` bytes/token placeholder needs to be revalidated against the actual GB10 attention configuration

### Mamba Cache Formats

This requires explicit measurement on GB10. Candidate storage/execution choices:

- FP32 cache / FP32 arithmetic
- FP16 cache / FP32 arithmetic
- FP16 cache with stochastic rounding on store
- INT8 cache with per-group scaling as an exploratory path, not a default assumption

Questions to answer:

- what is the decode-speed gain from reduced bandwidth?
- what is the quality / verbosity cost?
- what metadata or RNG state must be stored for stable replay and cache restore?

## Performance Hypotheses To Validate

These hypotheses are strong enough to guide implementation priority, but they still require measurement.

### H1: Device-Resident Dense Weights Are Mandatory

If dense weights remain host-backed and are copied per call, TTFT and decode results will be dominated by transfer overhead instead of actual compute behavior.

Expected action:

- upload dense weights once at startup
- retain device-resident descriptors in runtime-owned catalogs

### H2: KV Layout Should Follow cuDNN Paged Attention

For attention, the correct v1 data-structure target is likely:

- paged K/V containers
- page tables
- explicit sequence-length tensors

Expected action:

- design KV state around the cuDNN FE contract

### H3: Mamba Decode Speed Is Bandwidth-Limited

On GB10, Mamba recurrent-state traffic is likely to become one of the dominant decode constraints.

Expected action:

- benchmark cache format and conversion policy before locking state layout

### H4: CUDA Version Matters

The runtime should not treat CUDA `13.0` host behavior as representative if NVIDIA documents DGX Spark `NVFP4` / `MXFP8` improvements in newer stacks and if older stacks carry known `cuBLASLt` bugs on DGX Spark.

Expected action:

- standardize on CUDA `13.2+` before any serious BF16 / NVFP4 performance decisions

### H5: Speculative Decoding Is Likely The Highest-Leverage Decode Optimization After Quantization

On GB10, decode is deeply memory-bound. Speculative decoding trades idle compute for fewer full-model bandwidth passes and may deliver a larger decode-speed gain than kernel-level tuning alone.

Expected action:

- evaluate speculative decoding only after the baseline runtime is correct
- measure acceptance rate and end-to-end gain on representative prompts
- budget for private Mamba-state branches during draft verification

### H6: Operator Fusion Matters On GB10

On a bandwidth-limited UMA system, every unfused residual/norm/activation path pays extra global-memory traffic.

Expected action:

- audit the highest-ROI fusions first:
  - residual + RMSNorm
  - SwiGLU / GLU family
  - grouped GEMM + activation for MoE patterns
- prefer `cuDNN FE` runtime fusion or custom CUDA over Triton/CUTLASS SM100-specific paths

### H7: Mamba Cache Format Sensitivity Grows With Concurrency

Mamba state traffic is a smaller fraction of total decode bandwidth at batch `1`, but it grows materially with concurrency because weights are shared while recurrent state is per-request.

Expected action:

- benchmark Mamba cache formats at batch or concurrency points that match the service target:
  - `1`
  - `4`
  - `8`
- optionally include `16` as a stress point even if it exceeds the intended service target

## Required GB10 Performance Work Before Freezing Layouts

This should be treated as a hard gate.

### 0. Toolchain Upgrade And Pinning

Standardize the benchmark environment:

- CUDA `13.2+`
- matching cuBLASLt
- matching cuDNN Frontend development surface
- required driver / OS combination that actually exposes that stack on DGX Spark

Record:

- exact toolkit version
- exact driver version
- exact container / host path used for measurements

Treat this as the first gating item because:

- older stacks can invalidate BF16 / FP16 dense experiments
- older stacks understate NVFP4 performance on GB10

### 1. Dense GEMM Shape-Family Benchmarks

Measure representative real checkpoint shape families, not toy GEMMs.

For each shape family:

- BF16
- FP32 correctness baseline
- NVFP4 / MXFP8 where applicable

Measure:

- cold first-call latency
- hot repeated-call latency
- effective throughput
- workspace sensitivity
- heuristic reuse behavior

Current local harness status:

- implemented in `benchmarks/gb10_dense_gemm/gb10_dense_gemm_bench.cpp`
- reproducible runner script:
  - `benchmarks/gb10_dense_gemm/run_default_bench.sh`
- default cases derive from checkpoint-observed families such as:
  - `4096 x 4096` attention projection
  - `256 x 4096` attention KV projection
  - `5376 x 4096` / `4096 x 5376` shared expert projections
  - `18560 x 4096` / `4096 x 8192` Mamba projections
  - `2688 x 1024` / `1024 x 2688` MTP expert projections
- current smoke artifact:
  - [artifacts/benchmarks/gb10_dense_gemm_smoke.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_dense_gemm_smoke.json)
- current full-local artifact:
  - [artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.json)
- current local environment capture:
  - [artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.env.txt)
- current CUDA `13.2` full artifact:
  - [artifacts/benchmarks/gb10_dense_gemm_default_20260328T222445Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_dense_gemm_default_20260328T222445Z_cuda132.json)
- current CUDA `13.2` environment capture:
  - [artifacts/benchmarks/gb10_dense_gemm_default_20260328T222445Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_dense_gemm_default_20260328T222445Z_cuda132.env.txt)
- current limitation:
  - the harness measures the uploaded FP32 dense path first; BF16 and real checkpoint-driven NVFP4 execution still need explicit benchmark integration and activation-packing support

### 2. NVFP4 Contract Validation

Run targeted layout and descriptor experiments until we have at least one repeatable successful path for representative Blackwell shapes.

Current local status:

- implemented benchmark runner:
  - `benchmarks/gb10_nvfp4_contract/gb10_nvfp4_contract_bench.cu`
  - `benchmarks/gb10_nvfp4_contract/run_default_bench.sh`
- current full-local artifact:
  - [artifacts/benchmarks/gb10_nvfp4_contract_default_local_cuda13000.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_contract_default_local_cuda13000.json)
- current CUDA `13.2` full artifact:
  - [artifacts/benchmarks/gb10_nvfp4_contract_default_20260328T222445Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_contract_default_20260328T222445Z_cuda132.json)
- current local result:
  - the naive `row_nn_mk_kn` contract still fails with `CUBLAS_STATUS_NOT_SUPPORTED`
  - `row_nt_mk_nk` succeeds on representative local shapes and is the current runtime-facing choice
  - `col_tn_km_kn` also succeeds, but it is not the preferred runtime direction because current weight metadata and host-side interpretation are row-major

Still validate explicitly:

- base-pointer alignment against sample-backed `32`-byte expectations
- leading-dimension and shape constraints for the chosen contract
- exact scale-buffer layout and any required rearrangement before freezing the packed-format contract

Do not freeze:

- packed tensor layout
- scale-buffer layout
- auxiliary scale placement

until this succeeds on the intended pinned GB10 stack as well.

### 2B. Runtime-Side NVFP4 Executor Benchmarks

Measure the real runtime-side NVFP4 executor, not just the standalone contract probe.

Current local status:

- implemented benchmark runner:
  - `benchmarks/gb10_nvfp4_gemm/gb10_nvfp4_gemm_bench.cpp`
  - `benchmarks/gb10_nvfp4_gemm/run_default_bench.sh`
- implemented host-side activation packer used by that harness:
  - `runtime/include/nemotron/nvfp4_packing.h`
  - `runtime/src/backend/nvfp4_packing.cpp`
- implemented device-side activation packer and direct runtime execution path:
  - `runtime/include/nemotron/device_nvfp4_matrix.h`
  - `runtime/src/backend/device_nvfp4_matrix.cu`
  - `runtime/include/nemotron/nvfp4_gemm_runner.h`
  - `runtime/src/backend/nvfp4_gemm_runner.cpp`
  - current device packer now derives tensor scale from the actual device activation range rather than assuming `1.0`
  - both host-side and device-side packers now also support a caller-provided fixed tensor scale so the runtime can reproduce checkpoint `input_scale` values when needed
- implemented checkpoint-derived oracle tooling and fixture-backed smoke coverage:
  - `tools/oracle/dump_nvfp4_operator_fixture.py`
  - `tools/oracle/generate_nvfp4_operator_fixture.sh`
  - `testing/oracle/nvfp4_layer1_expert0_up_proj/`
  - `testing/backend/nvfp4_oracle_fixture_test.cpp`
- current CUDA `13.2` smoke artifact:
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T224447Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T224447Z_cuda132.json)
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T224447Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T224447Z_cuda132.env.txt)
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T225555Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T225555Z_cuda132.json)
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T225555Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T225555Z_cuda132.env.txt)
- current CUDA `13.2` full-family artifact:
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T225802Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T225802Z_cuda132.json)
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T225802Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T225802Z_cuda132.env.txt)
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231216Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231216Z_cuda132.json)
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231216Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231216Z_cuda132.env.txt)
- current CUDA `13.2` direct host-vs-device smoke artifacts:
  - host:
    - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231211Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231211Z_cuda132.json)
  - device:
    - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231440Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231440Z_cuda132.json)
    - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T233353Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T233353Z_cuda132.json)
    - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T233405Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T233405Z_cuda132.json)
- current smoke result:
  - earlier synthetic-activation smoke:
    - `attention_core_proj`, `m=64`, workspace `0`: `hot_mean_ms=0.041910`
    - `attention_core_proj`, `m=64`, workspace `4 MiB`: `hot_mean_ms=0.042011`
  - current host-packed-activation smoke:
    - `attention_core_proj`, `m=64`, workspace `0`: `hot_mean_ms=0.050630`
    - `attention_core_proj`, `m=64`, workspace `4 MiB`: `hot_mean_ms=0.049088`
  - current host-packed-activation staging cost at this shape:
    - workspace `0`:
      - `activation_pack_ms=5.915`
      - `activation_upload_ms=0.032`
    - workspace `4 MiB`:
      - `activation_pack_ms=5.851`
      - `activation_upload_ms=0.026`
  - conclusion from the current runtime path:
    - host-side FP32-to-NVFP4 activation packing now dominates activation upload cost by roughly two orders of magnitude for this smoke case
- current full-family result:
  - that same pattern held across the default case family sweep
  - representative `workspace=0`, `m=256` results:
    - `attention_core_proj`:
      - `activation_pack_ms=22.313`
      - `activation_upload_ms=0.030`
      - `hot_mean_ms=0.056`
    - `shared_expert_down`:
      - `activation_pack_ms=29.834`
      - `activation_upload_ms=0.157`
      - `hot_mean_ms=0.066`
    - `mtp_expert_up`:
      - `activation_pack_ms=5.557`
      - `activation_upload_ms=0.019`
      - `hot_mean_ms=0.029`
  - current implication:
    - on the measured runtime path, host activation packing is the next TTFT bottleneck to optimize before further NVFP4 GEMM tuning
- current device-staging result:
  - the device-side packer removes that host bottleneck on the measured runtime path
  - representative `workspace=0`, `m=256` results:
    - `attention_core_proj`:
      - `activation_pack_ms=0.038`
      - `activation_upload_ms=0.000`
      - `hot_mean_ms=0.057`
    - `shared_expert_down`:
      - `activation_pack_ms=0.158`
      - `activation_upload_ms=0.000`
      - `hot_mean_ms=0.067`
    - `mtp_expert_up`:
      - `activation_pack_ms=0.035`
      - `activation_upload_ms=0.000`
      - `hot_mean_ms=0.030`
  - direct comparison against the earlier host-packed full-family artifact:
    - `attention_core_proj`, `m=256`, `workspace=0`: about `587x` lower pack time
    - `shared_expert_down`, `m=256`, `workspace=0`: about `189x` lower pack time
    - `mtp_expert_up`, `m=256`, `workspace=0`: about `160x` lower pack time
    - `mtp_expert_down`, `m=256`, `workspace=0`: about `440x` lower pack time
  - direct smoke comparison at `attention_core_proj`, `m=64`:
    - host staging:
      - `activation_pack_ms=5.950`
      - `activation_upload_ms=0.120`
      - `hot_mean_ms=0.042`
    - device staging:
      - `activation_pack_ms=0.220`
      - `activation_upload_ms=0.000`
      - `hot_mean_ms=0.041`
  - current implication:
    - for the runtime-facing path, device-side activation staging is the correct default direction
    - host staging should remain as an explicit comparison mode, not the default benchmark path

Current limitation:

- the benchmark now supports both host-side and device-side activation staging, but the benchmark harness itself still starts from synthetic deterministic activations rather than checkpoint-driven operator outputs
- the device-side packer is still a narrow first kernel:
  - dynamic tensor-scale derivation now matches the host packer for bounded and large-value parity tests
  - but non-finite activation handling and wider checkpoint-driven activation validation still need to be added
- checkpoint-derived fixture coverage now includes:
  - `backbone.layers.1.mixer.experts.0.up_proj`
  - `backbone.layers.1.mixer.experts.0.down_proj`
  - `backbone.layers.1.mixer.shared_experts.down_proj`
- the current checkpoint-derived parity gap has been resolved for the current cuBLASLt path:
  - the runtime now keeps logical row-major NVFP4 scale buffers separate from padded-swizzled execution-layout scale buffers
  - the metadata-driven fixture-backed oracle test is now numerically gating across routed and shared expert paths
  - current validated results include:
    - routed `up_proj`: `max_abs_diff=9.53674e-07`, `mean_abs_diff=1.22253e-07`
    - routed `down_proj`: `max_abs_diff=1.39698e-09`, `mean_abs_diff=1.37535e-11`
    - shared `down_proj`: `max_abs_diff=1.49012e-08`, `mean_abs_diff=1.57186e-10`
- the independent CPU-side reconstructed reference is still useful as a diagnostic, but fixture-gold outputs are now the primary runtime gate
- the first device-staging benchmark after process start can still include one-time kernel/module initialization cost:
  - for example `attention_core_proj`, `m=64`, `workspace=0` measured `activation_pack_ms=27.463` on the first rerun and `0.281` on the immediate second rerun
  - if pack-time stability becomes a benchmark requirement, the harness should eventually report cold and warm pack cost separately
- current direct comparison status:
  - comparison tool:
    - `tools/benchmark_analysis/compare_gemm_benchmarks.py`
    - `benchmarks/gb10_gemm_compare/run_compare.sh`
  - current comparison artifacts:
    - [artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.json)
    - [artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.md)
  - current result on overlapping CUDA `13.2` cases:
    - matched comparisons: `45`
    - NVFP4 faster on pure GEMM hot latency: `39 / 45`
    - NVFP4 faster on runtime-facing hot latency: `35 / 45`
    - strongest measured win so far:
      - `shared_expert_up`, `m=256`, `workspace=0`
      - compute-only speedup: about `21.7x`
      - runtime-facing speedup: about `12.8x`
  - current implication:
    - attention and shared-expert families are already strong NVFP4 candidates on the runtime-facing GB10 path
    - the smallest `mtp_expert_up` / `mtp_expert_down` cases still lose once activation staging is included, especially at `m=1`

### 3. Paged Attention Benchmarks

Measure `cuDNN FE` attention for representative prefill and decode shapes.

Capture:

- block size choices
- page-table overhead
- hot/cold behavior
- prefill vs decode cost
- BF16 KV budget and bandwidth impact on GB10
- correctness and performance with GQA-style Q/KV head mismatch if present

### 4. Mamba Cache-Format Benchmarks

Measure recurrent-update and cache-load/store behavior for:

- FP32 cache
- FP16 cache
- FP16 cache with stochastic rounding
- INT8 per-group cache as an exploratory comparison

Capture:

- decode tok/s impact
- DRAM bandwidth pressure
- parity / verbosity impact on fixed prompts
- batch or concurrency scaling at:
  - `1`
  - `4`
  - `8`

### 5. Startup Residency Plan

Quantify what should be startup-resident on device for TTFT:

- embeddings
- dense projections
- shared experts
- routed experts if feasible
- scale buffers
- page tables / reusable metadata

This needs an explicit byte budget, not just a qualitative preference.

### 6. Model-Loading Benchmark

Benchmark startup loading mode on DGX Spark specifically:

- current `mmap` path
- eager read / copy path
- possible sequential-read hints

Capture:

- cold-start load time
- first-request TTFT impact after cold boot
- NVMe placement assumptions

Current harness status:

- implemented benchmark runner:
  - `benchmarks/gb10_loader_modes/gb10_loader_modes_bench.cpp`
  - `benchmarks/gb10_loader_modes/run_bench.sh`
- current smoke artifacts:
  - [artifacts/benchmarks/gb10_loader_modes_smoke.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_loader_modes_smoke.json)
  - [artifacts/benchmarks/gb10_loader_modes_20260328T192332Z.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_loader_modes_20260328T192332Z.json)
  - [artifacts/benchmarks/gb10_loader_modes_20260328T192332Z.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_loader_modes_20260328T192332Z.env.txt)
  - [artifacts/benchmarks/gb10_loader_modes_20260328T222550Z.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_loader_modes_20260328T222550Z.json)
  - [artifacts/benchmarks/gb10_loader_modes_20260328T222550Z.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_loader_modes_20260328T222550Z.env.txt)
- current limitation:
  - the smoke run used a synthetic tiny manifest only; the next meaningful run must target a real packed model manifest on DGX Spark

### 7. Speculative-Decoding Feasibility

Evaluate speculative decoding after the baseline runtime is correct.

Capture:

- acceptance rate on representative prompts
- net decode tok/s gain
- memory overhead for draft state
- Mamba branch-state overhead per speculative path

### 8. Operator-Fusion Audit

Benchmark the highest-ROI portable fusions for SM121:

- residual + RMSNorm
- SwiGLU / GLU family
- grouped GEMM + activation patterns for MoE

Primary mechanisms:

- `cuDNN FE` runtime fusion
- custom CUDA when `cuDNN FE` cannot express the needed pattern

### 9. OOM-Safety And Memory-Budget Validation

Validate the memory planner under the supported concurrency/context envelope.

Requirements:

- use `/proc/meminfo`-based signals as the primary allocatable-memory input
- keep a documented safety margin, on the order of `20 GiB`, rather than planning to the edge
- stress-test the maximum supported concurrency and context length without wedging the machine
- treat `cudaMemGetInfo` as advisory only on GB10

## Data-Structure Consequences

Until the GB10 performance pass is complete, treat these as provisional decisions:

- dense-weight storage objects
- NVFP4 packed-buffer and scale-buffer layouts
- Mamba cache storage format
- workspace reservation policy
- hot operator residency policy
- model-loading strategy (`mmap` vs eager read)

These are safer to freeze now:

- exact-prefix cache control-plane structure
- manifest-backed tensor metadata model
- per-tensor dtype/layout identity
- paged-attention direction for KV
- request-owned device tensor concept
- the row-major `A(m,k) / B(n,k) with transB = T` cuBLASLt contract for the current row-major NVFP4 execution path
- the need for separate logical raw NVFP4 scale buffers and padded-swizzled execution-layout NVFP4 scale buffers for the current cuBLASLt path
- the principle that Mamba prefix reuse must be modeled as checkpoint copy/fork cost rather than as KV-style refcounted page sharing

Current checkpoint-derived NVFP4 oracle status:

- routed-expert `up_proj` and `down_proj` are both now numerically gated against checkpoint-derived fixture gold outputs
- shared-expert `down_proj` is now also numerically gated against checkpoint-derived fixture gold output
- the remaining parity-expansion task is no longer “add a third fixture”; it is “add another distinct family or layer only if broader runtime parity coverage is needed”

Current Mamba cache benchmark status:

- **Decision reached**: use `FP16 + stochastic rounding` as the production optimized format; accept the known first-user-tail weakness; re-assess when production traces are available
- the benchmark harness covers synthetic (`clone`, `update`, `mamba_update`), layer-derived (`fixture_update`, `fixture_chain`), and oracle-derived (`fixture_trace`) paths
- `fp16_sr` latency overhead is small: roughly `+2-8%` over plain FP16 across synthetic and recurrence-like benchmarks
- the oracle-derived `fixture_trace` path replays real tokenized multi-turn conversations through layer-0 Mamba operands (hidden, dt, B, C) dumped from the real model
- a six-profile oracle-derived family was benchmarked: narrative, structured, tool-style, markdown-heavy, flat, and headerless variants
- `fp16_sr` beats plain `fp16` on mean drift at the majority of phase boundaries (assistant-decode: `1.9x-3.8x`, later user-tail: `1.5x-4.1x`)
- a known weakness exists in the first user-tail phase, where `fp16_sr` can perform slightly worse than plain `fp16`:
  - weakest observed ratio floor: `0.848` at `8` requests on the markdown-flat profile
  - both formats' absolute mean drift at that point is under `0.001`
- the weakness correlates with the `dt` operand-distribution (Pearson `r = -0.86` between ratio and share of steps above p75 dt), but the response to uniform dt scaling is non-monotonic, confirming it is a complex operand-distribution interaction rather than a simple amplitude effect
- serial investigation ruled out specific token patterns (bullets, headings, think-tags) and global `dt` threshold gating as fixes
- no hardware `cvt.rs.f16.f32` path on SM121 under CUDA `13.2`; software Philox SR is the baseline
- grouped INT8 rejected — quantize/dequantize overhead erases the bandwidth win

Key artifacts for future re-assessment:

- Latest guardrail report: [gb10_mamba_trace_guardrails_20260329T152925Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_mamba_trace_guardrails_20260329T152925Z.md)
- Latest operand report: [gb10_mamba_trace_operands_20260329T152925Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_mamba_trace_operands_20260329T152925Z.md)
- dt-bucket analysis: [gb10_mamba_trace_dt_buckets_20260329T163241Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_mamba_trace_dt_buckets_20260329T163241Z.md)
- dt-scale sweep: [gb10_mamba_trace_dt_scale_sweep_20260329T164125Z_cuda132.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_mamba_trace_dt_scale_sweep_20260329T164125Z_cuda132.md)
- Threshold gating rejection: [gb10_mamba_dt_threshold_0p65_family_20260329T165952Z_cuda132.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/gb10_mamba_dt_threshold_0p65_family_20260329T165952Z_cuda132.md)
- Trace fixture generator: [dump_mamba_target_trace_fixture.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/tools/oracle/dump_mamba_target_trace_fixture.py)
- Benchmark supports `--dt-threshold` for future selective-gating experiments

## Recommended Immediate Next Step

The Mamba cache-format decision is now resolved. The next implementation steps should be:

1. Rerun the loader benchmark on a real packed manifest on DGX Spark instead of the synthetic smoke fixture
2. Refine the paged-attention benchmark case set and staging model around the service envelope now that the first full sweep exists
3. Add wider attention correctness cases only where a new runtime risk justifies them
4. Add another checkpoint-derived NVFP4 oracle only if a new operator family or parity risk justifies it
5. When production traces become available, re-run the Mamba `fixture_trace` path against them to validate the first-user-tail weakness under real serving load

Only after that should we lock more forward-path and memory-layout decisions.

## Exit Criteria

This performance-planning sub-project is complete when:

- we have repeatable GB10 measurements for the main dense shape families
- we have at least one validated NVFP4 `cuBLASLt` path for representative shapes
- that NVFP4 result has been confirmed on the pinned GB10 toolchain
- ~~we have a measured decision on Mamba cache format for decode~~ **done**: `FP16 + stochastic rounding`, with accepted first-user-tail weakness; re-assess on production traces
- we have a documented startup-residency policy for hot weights
- we have an OOM-safe memory budget validated by stress testing on the supported service envelope
- the runtime's major in-memory operator/state layouts can be justified by GB10 data rather than by generic CUDA assumptions
