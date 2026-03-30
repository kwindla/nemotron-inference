# GB10 Performance Progress

## Purpose

This file is the execution history for the GB10 optimization sub-project.

Use it to record:

- what optimization work was added
- what measurements were taken
- what decisions changed because of those measurements
- what remains unresolved before low-level layout or operator choices are frozen

This is complementary to:

- [PROGRESS.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/PROGRESS.md) for the full project history
- [gb10_performance_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gb10_performance_plan.md) for the optimization plan and open questions

## Logging Rule

For future GB10 optimization work, add an entry here whenever one of these happens:

- a new benchmark or probe is added
- a new measurement artifact materially changes a design decision
- a toolchain or backend constraint changes
- an optimization hypothesis is rejected or confirmed

Each entry should include:

- goal
- implementation or measurement scope
- artifacts or code touched
- result
- decision impact
- remaining risk

## 2026-03-28 - GB10 Baseline And Constraint Discovery

### Goal

Establish the real GB10 hardware, checkpoint, and memory constraints before freezing runtime layout decisions.

### Scope

- preflight checkpoint inspection
- memory-budget modeling
- initial backend feasibility probes
- first baseline smoke measurements

### Key Artifacts

- [artifacts/preflight/preflight_summary.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/preflight/preflight_summary.md)
- [artifacts/preflight/checkpoint_report.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/preflight/checkpoint_report.json)
- [artifacts/preflight/memory_budget_report.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/preflight/memory_budget_report.md)
- [artifacts/preflight/cublaslt_probe.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/preflight/cublaslt_probe.json)
- [artifacts/preflight/vllm_smoke.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/preflight/vllm_smoke.json)

### Result

- confirmed GB10 / DGX Spark as a low-bandwidth UMA target
- confirmed the real checkpoint mix of Mamba, attention, and MoE layers
- confirmed that naive dense Mamba checkpointing at many boundaries would explode memory use
- confirmed that the first naive FP4 `cuBLASLt` attempt was not enough to establish a valid runtime contract

### Decision Impact

- keep weight and state layouts provisional until GB10 measurements exist
- treat `cuBLASLt` and `cuDNN FE` as hard runtime constraints on GB10
- optimize for full-state conversation heads and shared roots, not dense Mamba checkpoints

### Remaining Risk

- no GB10-specific operator benchmarking yet
- no validated mixed-precision runtime path yet

## 2026-03-28 - Dense GEMM Benchmarking And Startup Residency Direction

### Goal

Benchmark the first real dense runtime path on GB10 and use it to inform startup-residency policy.

### Scope

- startup-uploaded dense weights
- dense shape-family benchmark harness
- local CUDA `13.0` baselines
- later CUDA `13.2` reruns
- first residency-policy write-up

### Key Code And Docs

- [benchmarks/gb10_dense_gemm/gb10_dense_gemm_bench.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_dense_gemm/gb10_dense_gemm_bench.cpp)
- [benchmarks/gb10_dense_gemm/run_default_bench.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_dense_gemm/run_default_bench.sh)
- [docs/startup_residency_policy.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/startup_residency_policy.md)

### Key Artifacts

- [artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.json)
- [artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.env.txt)

### Result

- workspace sensitivity was real on GB10 for several important dense families
- dense weight upload had to become startup-owned rather than per-call
- startup residency became a measured policy problem, not an assumption

### Decision Impact

- keep hot dense backbone weights resident
- do not design the runtime around repeated weight copies
- keep family-specific residency decisions open until broader measurements are in

### Remaining Risk

- dense results alone do not determine NVFP4, attention, or Mamba policy

## 2026-03-28 - CUDA 13.2 Stack Pinning And NVFP4 Contract Validation

### Goal

Move GB10 mixed-precision work onto the intended CUDA stack and validate a usable NVFP4 `cuBLASLt` contract.

### Scope

- CUDA `13.2` compat-path integration
- NVFP4 contract-validation harness
- first real runtime-side NVFP4 executor

### Key Code And Artifacts

- [benchmarks/gb10_nvfp4_contract/gb10_nvfp4_contract_bench.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_nvfp4_contract/gb10_nvfp4_contract_bench.cu)
- [runtime/src/backend/nvfp4_gemm_runner.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/src/backend/nvfp4_gemm_runner.cpp)
- [artifacts/benchmarks/gb10_nvfp4_contract_default_20260328T222445Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_nvfp4_contract_default_20260328T222445Z_cuda132.json)

### Result

- the runtime converged on the current row-major `A(m,k)` plus row-major `B(n,k)` with `transB=T` contract for the first NVFP4 path
- CUDA `13.2` became the required GB10 measurement stack

### Decision Impact

- treat earlier CUDA `13.0` results as historical only
- keep the current NVFP4 `cuBLASLt` contract as the runtime default until disproven by later GB10 data

### Remaining Risk

- contract validation alone does not prove activation staging or checkpoint parity

## 2026-03-28 - NVFP4 Activation Staging On GB10

### Goal

Determine whether NVFP4 activation staging is a TTFT bottleneck and whether it belongs on host or device.

### Scope

- host-side FP32-to-NVFP4 packer
- device-side FP32-to-NVFP4 packer
- runtime-side NVFP4 GEMM benchmark harness
- dynamic tensor-scale derivation

### Key Code

- [runtime/src/backend/nvfp4_packing.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/src/backend/nvfp4_packing.cpp)
- [runtime/src/backend/device_nvfp4_matrix.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/src/backend/device_nvfp4_matrix.cu)
- [benchmarks/gb10_nvfp4_gemm/gb10_nvfp4_gemm_bench.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_nvfp4_gemm/gb10_nvfp4_gemm_bench.cpp)

### Key Artifacts

- host/device comparison:
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231211Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231211Z_cuda132.json)
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231440Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231440Z_cuda132.json)
- full device-staged family sweep:
  - [artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231216Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_nvfp4_gemm_default_20260328T231216Z_cuda132.json)

### Result

- host-side activation packing was a major TTFT bottleneck
- device-side staging reduced activation pack cost by orders of magnitude on representative shapes
- dynamic tensor-scale derivation closed the first large-value correctness gap in the device packer

### Decision Impact

- device-side activation staging is the runtime default direction on GB10
- host-side staging remains only as a comparison mode

### Remaining Risk

- the benchmark path still needed checkpoint-derived rather than synthetic activations
- cold-vs-warm pack timing remained visible and may still matter

## 2026-03-28 To 2026-03-29 - Checkpoint-Derived NVFP4 Fixture, Execution-Scale Layout Fix, And Numerical Oracle

### Goal

Move NVFP4 correctness work beyond synthetic inputs by exercising a real checkpoint-derived expert operator with the runtime’s mixed-precision path.

### Scope

- fixed activation tensor-scale support in host and device packers
- checkpoint-derived oracle-fixture generation
- fixture-backed C++ smoke test

### Key Code And Artifacts

- [tools/oracle/dump_nvfp4_operator_fixture.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/dump_nvfp4_operator_fixture.py)
- [tools/oracle/generate_nvfp4_operator_fixture.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/generate_nvfp4_operator_fixture.sh)
- [testing/backend/nvfp4_oracle_fixture_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/backend/nvfp4_oracle_fixture_test.cpp)
- [testing/oracle/nvfp4_layer1_expert0_up_proj/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/nvfp4_layer1_expert0_up_proj/metadata.json)

### Result

- the runtime can now reproduce checkpoint-provided activation `input_scale`
- the first real expert fixture targets `backbone.layers.1.mixer.experts.0.up_proj`
- the runtime now keeps logical row-major NVFP4 scale buffers separate from padded-swizzled execution-layout scale buffers for `cuBLASLt`
- the fixture-backed oracle is now numerically gating with `max_abs_diff=1.90735e-06` and `mean_abs_diff=2.22149e-07`
- the checkpoint-derived independent reference is no longer treated as the unresolved problem for this first expert fixture

### Decision Impact

- keep the dual raw/execution NVFP4 scale representation for the cuBLASLt path
- treat the first checkpoint-derived NVFP4 oracle as a real correctness gate, not smoke coverage

### Remaining Risk

- this closes only the first expert fixture and the current cuBLASLt path
- broader checkpoint-derived oracle coverage is still needed before declaring NVFP4 parity solved across operator families

## 2026-03-29 - Metadata-Driven Multi-Fixture NVFP4 Oracle Coverage

### Goal

Extend the first checkpoint-derived NVFP4 oracle into multi-fixture coverage so routed-expert parity is not anchored on a single operator.

### Scope

- metadata-driven oracle-fixture discovery in the C++ test
- generalized fixture generation for `up_proj` and `down_proj`
- second checkpoint-derived routed-expert fixture

### Key Code And Artifacts

- [testing/backend/nvfp4_oracle_fixture_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/backend/nvfp4_oracle_fixture_test.cpp)
- [tools/oracle/dump_nvfp4_operator_fixture.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/dump_nvfp4_operator_fixture.py)
- [tools/oracle/generate_nvfp4_operator_fixture.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/generate_nvfp4_operator_fixture.sh)
- [testing/oracle/nvfp4_layer1_expert0_up_proj/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/nvfp4_layer1_expert0_up_proj/metadata.json)
- [testing/oracle/nvfp4_layer1_expert0_down_proj/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/nvfp4_layer1_expert0_down_proj/metadata.json)

### Result

- the oracle test now discovers fixture directories under `testing/oracle/` instead of being hardcoded to one shape
- routed-expert `up_proj` and `down_proj` are both now numerically gating against checkpoint-derived fixture gold outputs
- current runtime diagnostics:
  - `up_proj`: `max_abs_diff=9.53674e-07`, `mean_abs_diff=1.22253e-07`
  - `down_proj`: `max_abs_diff=1.39698e-09`, `mean_abs_diff=1.37535e-11`

### Decision Impact

- keep fixture-generated checkpoint outputs as the primary parity oracle for runtime NVFP4 execution
- treat the C++ host-side reconstructed reference as diagnostic-only unless it becomes a runtime issue
- future NVFP4 operator-family expansion should add new fixture directories rather than new test binaries

### Remaining Risk

- routed-expert parity is stronger now, but the oracle still needs at least one additional non-identical family, such as a shared expert
- the host-side reconstructed reference still drifts slightly from the `up_proj` fixture gold, with about `1.67e-06` max absolute difference, although that is well below the runtime gate

## 2026-03-29 - Third NVFP4 Oracle Fixture From Shared-Expert Family

### Goal

Add a checkpoint-derived NVFP4 oracle from a meaningfully different family so parity is no longer limited to routed experts.

### Scope

- extend fixture generation to support `shared_down_proj`
- generate a shared-expert fixture from the layer-1 checkpoint path
- validate that the existing metadata-driven oracle test gates it alongside the routed-expert fixtures

### Key Code And Artifacts

- [tools/oracle/dump_nvfp4_operator_fixture.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/dump_nvfp4_operator_fixture.py)
- [tools/oracle/generate_nvfp4_operator_fixture.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/generate_nvfp4_operator_fixture.sh)
- [testing/oracle/nvfp4_layer1_shared_down_proj/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/nvfp4_layer1_shared_down_proj/metadata.json)
- [testing/backend/nvfp4_oracle_fixture_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/backend/nvfp4_oracle_fixture_test.cpp)

### Result

- the third fixture targets `backbone.layers.1.mixer.shared_experts.down_proj`
- the generator now reconstructs its activation path as:
  - layer-1 normalized hidden states
  - `shared_experts.up_proj` in FP8
  - configured hidden activation
  - `shared_experts.down_proj` as the NVFP4 target
- the runtime remains numerically tight on this larger shared-expert path:
  - `max_abs_diff=1.49012e-08`
  - `mean_abs_diff=1.57186e-10`

### Decision Impact

- checkpoint-derived NVFP4 parity now spans both routed and shared expert families
- the immediate next GB10 optimization step should shift toward benchmark comparison and loader/attention work rather than continuing to add fixtures by default

### Remaining Risk

- parity coverage is stronger, but still concentrated in layer 1 and MoE-family operators
- more fixtures should be added only when a new family or suspected runtime risk justifies them

## 2026-03-29 - First Direct NVFP4 vs FP32 Comparison Report

### Goal

Turn the dense and NVFP4 benchmark artifacts into one case-by-case GB10 comparison so runtime policy decisions can be based on measured overlap rather than separate benchmark logs.

### Scope

- comparison tool for dense and NVFP4 benchmark JSON artifacts
- JSON and Markdown artifact output
- first report on the current CUDA `13.2` runtime-facing device-staged path

### Key Code And Artifacts

- [tools/benchmark_analysis/compare_gemm_benchmarks.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/benchmark_analysis/compare_gemm_benchmarks.py)
- [benchmarks/gb10_gemm_compare/run_compare.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_gemm_compare/run_compare.sh)
- [artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.json)
- [artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.md)

### Result

- matched comparisons: `45`
- NVFP4 faster on pure GEMM hot latency: `39 / 45`
- NVFP4 faster on runtime-facing hot latency: `35 / 45`
- strongest measured win:
  - `shared_expert_up`, `m=256`, `workspace=0`
  - compute-only speedup: about `21.7x`
  - runtime-facing speedup: about `12.8x`
- strongest losses remain concentrated in the smallest MTP cases, especially `m=1`

### Decision Impact

- attention and shared-expert families are already strong NVFP4 candidates on the current GB10 runtime-facing path
- MTP-family policy should stay conditional rather than blanket NVFP4 until we decide whether those small-batch losses matter for the service workload
- the next optimization step should move from raw benchmarking to explicit execution-policy selection for overlapping families

### Remaining Risk

- the comparison currently covers only the overlap between the existing dense and NVFP4 harnesses
- dense-only families such as attention KV and Mamba projections still need their own policy work

## 2026-03-29 - GB10 Execution Policy And Paged-Attention Kickoff

### Goal

Freeze the current GB10 operator-format decision into a short policy note, then begin the paged-attention path with concrete KV-page and page-table contracts.

### Scope

- short execution-policy note from the current comparison artifact
- first paged-KV geometry and allocation layer
- first padded page-table builder for later cuDNN FE attention integration

### Key Code And Artifacts

- [docs/gb10_execution_policy.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gb10_execution_policy.md)
- [runtime/include/nemotron/paged_kv_cache.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/include/nemotron/paged_kv_cache.h)
- [runtime/include/nemotron/paged_attention_plan.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/include/nemotron/paged_attention_plan.h)
- [testing/attention/paged_kv_cache_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/attention/paged_kv_cache_test.cpp)
- [testing/attention/paged_attention_plan_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/attention/paged_attention_plan_test.cpp)

### Result

- the short execution-policy note now captures the current measured stance:
  - NVFP4 is already favored for the measured attention-core and shared-expert families
  - small MTP cases remain conditional
  - Mamba projections and attention-KV remain unmeasured in the mixed-precision runtime path
- the runtime now has:
  - KV bytes-per-token and bytes-per-page derivation
  - a fixed-page KV arena with per-layer ownership
  - padded `INT32` page-table planning for mixed sequence lengths

### Decision Impact

- the project can start attention integration from a concrete KV/page-table contract instead of inventing it during backend work
- at the time of this stage, the remaining attention blocker was the missing cuDNN FE development surface on the machine

### Remaining Risk

- this stage was scaffolding only; there was not yet an executable cuDNN FE SDPA path
- at the time of this stage, backend attention work remained blocked on cuDNN header/library availability

## 2026-03-29 - First Executable BF16 cuDNN FE Paged Attention Path

### Goal

Move paged attention from control-plane scaffolding into a real executable backend path on GB10.

### Scope

- add runtime cuDNN handle ownership
- add a backend-specific paged-SDPA plan builder on top of `PagedAttentionBatchPlan`
- execute a decode-style BF16 paged-attention case through cuDNN FE and validate it against a CPU reference

### Key Code And Artifacts

- [runtime/include/nemotron/cudnn_handle.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/include/nemotron/cudnn_handle.h)
- [runtime/include/nemotron/cudnn_paged_attention.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/include/nemotron/cudnn_paged_attention.h)
- [runtime/src/backend/cudnn_handle.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/src/backend/cudnn_handle.cpp)
- [runtime/src/backend/cudnn_paged_attention.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/src/backend/cudnn_paged_attention.cpp)
- [testing/backend/cudnn_paged_attention_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/backend/cudnn_paged_attention_test.cpp)

### Result

- the runtime now builds a cuDNN FE paged-SDPA graph directly from the existing KV/page-table scaffolding
- the first execution path is intentionally narrow:
  - BF16 I/O
  - padded Q tensor
  - explicit `INT32` `seq_len_q`, `seq_len_kv`, and paged K/V tables
  - decode-style shape coverage first
  - no statistics output requirement
- the executable smoke test passed against a CPU reference on real device execution

### Decision Impact

- paged attention is no longer a planning-only subsystem
- the next GB10 attention step should be measurement and surface widening, not a second backend experiment

### Remaining Risk

- current execution coverage is decode-style and non-causal
- prefill, causal masking, and benchmark harness work still need to be added before attention policy is fully grounded

## 2026-03-29 - First GB10 Paged-Attention Benchmark Harness

### Goal

Turn the working BF16 cuDNN FE attention backend into a repeatable GB10 benchmark surface and capture the first decode/prefill smoke artifact.

### Scope

- dedicated paged-attention benchmark executable
- default decode and prefill case set aligned with the runtime's GB10 attention geometry
- wrapper script with environment capture and JSON artifact output
- first smoke run on one decode and one prefill case

### Key Code And Artifacts

- [benchmarks/gb10_paged_attention/gb10_paged_attention_bench.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_paged_attention/gb10_paged_attention_bench.cpp)
- [benchmarks/gb10_paged_attention/run_default_bench.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_paged_attention/run_default_bench.sh)
- [artifacts/benchmarks/gb10_paged_attention_20260329T031126Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_paged_attention_20260329T031126Z_cuda132.json)
- [artifacts/benchmarks/gb10_paged_attention_20260329T031126Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_paged_attention_20260329T031126Z_cuda132.env.txt)

### Result

- the runtime now has a dedicated attention benchmark harness alongside the dense, NVFP4, and loader harnesses
- the first smoke artifact covered:
  - `decode_b1_kv64k`
  - `prefill_b1_q256_kv8k`
- the main early signal from the smoke artifact is structural:
  - execution latency is extremely small once the plan is built and inputs are resident
  - plan build and input upload dominate these one-case smoke measurements

### Decision Impact

- the next attention question is no longer “can the backend run?” It is “which decode/prefill shapes matter enough to benchmark and optimize first?”
- GB10 attention work can now use the same artifact discipline as the GEMM and loader sub-projects

### Remaining Risk

- the current smoke artifact is not a full sweep and should not yet be treated as a final policy input
- causal and prefill behavior are present in the benchmark path, but correctness gating for those wider cases still needs to be added

## 2026-03-29 - Wider Attention Correctness Gate And First Full Paged-Attention Sweep

### Goal

Close the gap between the first attention smoke path and the benchmark/default service cases by adding broader correctness coverage and running the full benchmark suite.

### Scope

- widen the CPU-reference-gated cuDNN attention test beyond non-causal decode
- run the full default paged-attention benchmark suite
- record the first full decode/prefill GB10 attention artifact

### Key Code And Artifacts

- [testing/backend/cudnn_paged_attention_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/backend/cudnn_paged_attention_test.cpp)
- [artifacts/benchmarks/gb10_paged_attention_20260329T031635Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_paged_attention_20260329T031635Z_cuda132.json)
- [artifacts/benchmarks/gb10_paged_attention_20260329T031635Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_paged_attention_20260329T031635Z_cuda132.env.txt)

### Result

- the cuDNN attention test now covers:
  - decode-style non-causal reference parity
  - small causal prefill reference parity
- the first full benchmark artifact now covers all current default cases:
  - `decode_b1_kv64k`
  - `decode_b8_kv64k`
  - `prefill_b1_q256_kv8k`
  - `prefill_b1_q512_kv32k`

### Decision Impact

- the attention backend is now both benchmarked and correctness-gated on the causal path that the current benchmark defaults actually use
- the dominant signal from the benchmark is now clear:
  - hot attention execution is very small in these cases
  - plan build and host-to-device staging dominate the harness-level measurement

### Remaining Risk

- the current CPU reference only covers smaller causal/prefill cases, not the largest benchmark shapes
- the benchmark harness still stages full synthetic inputs per run, so it should not yet be interpreted as a pure serving-path TTFT number

## 2026-03-29 - First GB10 Mamba Cache-Format Benchmark Harness

### Goal

Establish the first repeatable GB10 measurement surface for Mamba cache-format tradeoffs before freezing the decode-cache policy.

### Scope

- add a dedicated Mamba cache benchmark executable
- measure clone and synthetic update cost for FP32, FP16, and exploratory grouped-INT8 formats
- sweep service-relevant concurrency points
- emit JSON artifacts with effective bandwidth estimates

### Key Code And Artifacts

- [benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu)
- [benchmarks/gb10_mamba_cache/run_default_bench.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_cache/run_default_bench.sh)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T032215Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T032215Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T032501Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T032501Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T032501Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T032501Z_cuda132.env.txt)

### Result

- the runtime now has a dedicated GB10 Mamba cache benchmark alongside the dense, NVFP4, attention, and loader harnesses
- the first artifact was a focused smoke run at concurrency `1` and `8`
- the second artifact is the first full default sweep at concurrency `1`, `4`, and `8`
- the benchmark reports both latency and an approximate effective bandwidth derived from the modeled state traffic

### Decision Impact

- FP32 and FP16 clone/update paths both sustain roughly `226-243 GB/s`, so the current synthetic kernels are already behaving like a bandwidth-bound workload on GB10
- FP16 improves latency mainly by moving fewer bytes, not by unlocking a meaningfully higher bandwidth regime
- grouped-INT8 remains exploratory:
  - grouped-INT8 clone tracks similar copy bandwidth
  - grouped-INT8 update drops to roughly `155-168 GB/s`, so extra quantize/dequantize work currently outweighs the traffic reduction

### Remaining Risk

- the current update kernels are synthetic bandwidth probes, not the final Mamba recurrence kernels
- there is still no FP16 plus stochastic-rounding update path in the benchmark, which is the most relevant next comparison against the FP32 baseline
- the real-manifest loader benchmark is still pending because there is no packed production manifest in the workspace

## 2026-03-29 - FP16 Plus Stochastic-Rounding Mamba Cache Benchmark Path

### Goal

Add the first benchmarked `FP16 + stochastic rounding` update path so the Mamba cache study measures the actual production candidate instead of only plain FP16.

### Scope

- extend the benchmark format set with `fp16_sr`
- implement deterministic counter-based Philox-style stochastic rounding for the synthetic FP16 update path
- record the stochastic-rounding seed and round count in the benchmark artifact
- rerun smoke and full default sweeps on the pinned CUDA `13.2` stack

### Key Code And Artifacts

- [benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T033053Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T033053Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T033101Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T033101Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T033101Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T033101Z_cuda132.env.txt)

### Result

- the benchmark now records four cache formats:
  - `fp32`
  - `fp16`
  - `fp16_sr`
  - exploratory `int8_group`
- the `fp16_sr` path uses a deterministic counter-based RNG with:
  - seed `5639999111463849804`
  - `5` Philox rounds
- the first full default artifact shows that `fp16_sr` remains close to plain FP16 on the current synthetic update kernel

### Decision Impact

- `fp16_sr` is now a measured candidate rather than a paper-only recommendation
- the current overhead versus plain FP16 update is modest on GB10:
  - about `+1.9%` at concurrency `1`
  - about `+7.6%` at concurrency `4`
  - about `+4.9%` at concurrency `8`
- that keeps `FP16 + stochastic rounding` as the leading optimized-cache candidate, while grouped INT8 still looks too expensive on update

### Remaining Risk

- this is still a software Philox-style rounding path in a synthetic update benchmark, not yet a recurrence-aware Mamba kernel
- SM121 hardware stochastic-rounding instruction availability is still unverified
- the benchmark still measures performance only; it does not yet characterize long-step numerical drift relative to an FP32 recurrent baseline

## 2026-03-29 - SM121 Stochastic-Rounding PTX Probe

### Goal

Resolve the narrow toolchain question of whether `sm_121` exposes a direct PTX `fp32 -> fp16` stochastic-rounding conversion path under the pinned CUDA `13.2` stack.

### Scope

- add a standalone PTX probe script
- compile a control `cvt.rn.f16.f32` case with `ptxas`
- compile a candidate `cvt.rs.f16.f32` case with `ptxas`
- capture the result as a machine-readable artifact

### Key Code And Artifacts

- [tools/probes/sm121_stochastic_rounding_probe.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/probes/sm121_stochastic_rounding_probe.py)
- [benchmarks/gb10_stochastic_rounding_probe/run_probe.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_stochastic_rounding_probe/run_probe.sh)
- [artifacts/benchmarks/gb10_stochastic_rounding_probe_20260329T034157Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_stochastic_rounding_probe_20260329T034157Z_cuda132.json)
- [artifacts/benchmarks/gb10_stochastic_rounding_probe_20260329T034157Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_stochastic_rounding_probe_20260329T034157Z_cuda132.env.txt)

### Result

- `ptxas` accepted the control `cvt.rn.f16.f32` case for `sm_121`
- `ptxas` rejected `cvt.rs.f16.f32` for `sm_121`
- the probe therefore did not find a direct PTX `fp32 -> fp16` stochastic-rounding conversion path on this stack

### Decision Impact

- the current software Philox-style stochastic-rounding path should be treated as the baseline implementation on GB10
- the project should not assume a hidden direct PTX `cvt.rs.f16.f32` fast path exists on `sm_121`

### Remaining Risk

- this probe only answers the direct PTX `cvt.rs.f16.f32` question
- if SM121 exposes some other hardware-assisted stochastic-rounding path, it would need a different probe and a different implementation strategy

## 2026-03-29 - Recurrence-Like Mamba Update Benchmark Path

### Goal

Move the Mamba cache study one step closer to the actual recurrence workload by adding a state-mixing update path instead of relying only on the earlier synthetic per-element update.

### Scope

- extend the benchmark with a new `mamba_update` operation
- make `mamba_update` read both the current state element and a neighboring state element before writing the next state
- run smoke and full default artifacts with `fp32`, `fp16`, `fp16_sr`, and exploratory `int8_group`

### Key Code And Artifacts

- [benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T034103Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T034103Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T034115Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T034115Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T034115Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T034115Z_cuda132.env.txt)

### Result

- the benchmark now has three operation classes:
  - `clone`
  - `update`
  - `mamba_update`
- `mamba_update` keeps `fp16_sr` close to plain `fp16`:
  - about `+4.4%` latency at concurrency `1`
  - about `+4.6%` latency at concurrency `4`
  - about `+1.2%` latency at concurrency `8`

### Decision Impact

- the stronger result is now consistent across both update styles:
  - `FP16 + stochastic rounding` remains close enough to plain `fp16` to stay the lead optimized-cache candidate
- grouped INT8 still does not overtake the FP16-family paths on the more recurrence-like operation

### Remaining Risk

- `mamba_update` is still a recurrence-like synthetic kernel, not a full Mamba layer kernel
- the benchmark's weighted traffic score can exceed physical DRAM bandwidth because the neighbor-read path can reuse cache lines; it should not be interpreted as literal LPDDR throughput

## 2026-03-29 - Layer-Derived Mamba Decode-Update Oracle Fixture

### Goal

Add a numerically checked Mamba update path rooted in real checkpoint layer tensors instead of relying only on synthetic or recurrence-like benchmark kernels.

### Scope

- dump a decode-step oracle fixture for `backbone.layers.0.mixer`
- use real checkpoint tensors for `A_log`, `dt_bias`, and `D`
- generate deterministic real-shape decode operands and gold outputs
- add a CUDA test that replays the decode-step update and checks next-state and output parity

### Key Code And Artifacts

- [tools/oracle/dump_mamba_update_fixture.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/dump_mamba_update_fixture.py)
- [tools/oracle/generate_mamba_update_fixture.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/generate_mamba_update_fixture.sh)
- [testing/backend/mamba_update_fixture_test.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/backend/mamba_update_fixture_test.cu)
- [testing/oracle/mamba_layer0_decode_update/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_layer0_decode_update/metadata.json)

### Result

- the project now has a layer-derived Mamba decode-update oracle fixture
- the first CUDA gate replays the fixture in FP32 and matches both:
  - `expected_next_state_fp32.bin`
  - `expected_output_fp32.bin`
- the first test run came in at:
  - `next_state_diff=5.96046e-08`
  - `output_diff=2.86102e-06`

### Decision Impact

- the next Mamba performance step can now benchmark a real layer-shaped decode update instead of only synthetic traffic models
- the project now has a correctness anchor for later `fp16` / `fp16_sr` fixture-path experiments

### Remaining Risk

- the current fixture uses real layer tensors for `A_log`, `dt_bias`, and `D`, but deterministic synthetic decode operands for `hidden`, `dt_pre`, `B`, `C`, and initial `ssm_state`
- there is still no fixture-driven multi-format benchmark path yet

## 2026-03-29 - Layer-Derived Mamba Fixture Benchmark Path

### Goal

Turn the layer-derived Mamba fixture into a real GB10 benchmark mode so cache-format work can be judged on both latency and one-step drift instead of on synthetic traffic alone.

### Scope

- add a `fixture_update` operation to the GB10 Mamba benchmark
- support `fp32`, `fp16`, and `fp16_sr`
- reset to the same initial fixture state before each timed iteration so all timed runs compare to the same oracle gold tensors
- emit one-step next-state and output drift metrics in the JSON artifact

### Key Code And Artifacts

- [benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T041811Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T041811Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T041842Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T041842Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T041842Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T041842Z_cuda132.env.txt)

### Result

- the benchmark now has four operation classes:
  - `clone`
  - `update`
  - `mamba_update`
  - `fixture_update`
- on the real layer-derived one-step path:
  - `fp32` stays effectively exact, with `max_output_abs_diff=0.000003`
  - `fp16` runs at `0.034 / 0.106 / 0.240 ms` for `1 / 4 / 8` requests and stays at `max_output_abs_diff=0.001579`
  - `fp16_sr` runs at `0.040 / 0.125 / 0.271 ms` for `1 / 4 / 8` requests and stays in the same drift range, with `max_output_abs_diff=0.001172` at `1 / 4` requests and `0.001709` at `8`

### Decision Impact

- the project now has a real layer-shaped Mamba benchmark mode, not just a standalone CUDA gate
- `fp16_sr` remains in the lead decision set:
  - it is slower than plain `fp16` on the one-step fixture path, but still fast enough to stay viable on GB10
  - it can now be judged against real one-step drift numbers instead of only synthetic bandwidth proxies
- grouped `INT8` remains intentionally excluded from this path until there is a stronger reason to pay the added complexity

### Remaining Risk

- this is still a one-step decode-update study; it does not yet characterize accumulated drift over long decode chains
- the fixture still uses deterministic synthetic decode operands for some tensors, even though the core layer tensors come from the checkpoint
- for `fp16_sr`, one-seed one-step drift is not enough to validate the stochastic-rounding story; the next study should track cross-seed variance as well as max drift

## 2026-03-29 - Multi-Step Mamba Fixture Chain And Cross-Seed Drift

### Goal

Turn the Mamba fixture study into a real accumulated-drift benchmark and check whether `fp16_sr` stays bounded across multiple fixed seeds rather than only on a single one-step run.

### Scope

- add a `fixture_chain` operation to the GB10 Mamba benchmark
- chain the layer-derived decode update for configurable step counts
- keep `fixture_update` as the one-step anchor
- aggregate `fp16_sr` drift across several fixed seeds
- emit per-result max / mean / RMS drift plus cross-seed standard deviations

### Key Code And Artifacts

- [benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T043419Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T043419Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T043453Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T043453Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T043453Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T043453Z_cuda132.env.txt)

### Result

- the benchmark now has five operation classes:
  - `clone`
  - `update`
  - `mamba_update`
  - `fixture_update`
  - `fixture_chain`
- at `128` chained steps, `fp16_sr` is materially better than plain `fp16` on accumulated output drift:
  - `1` request:
    - `fp16 mean_output_abs_diff=0.003940466`
    - `fp16_sr mean_output_abs_diff=0.000308270`
  - `8` requests:
    - `fp16 mean_output_abs_diff=0.003940466`
    - `fp16_sr mean_output_abs_diff=0.000314088`
- the `fp16_sr` cross-seed spread stays small on the same `128`-step study:
  - `1` request:
    - `stddev_mean_output_abs_diff=0.000002214`
    - `stddev_rms_output_abs_diff=0.000015082`
  - `8` requests:
    - `stddev_mean_output_abs_diff=0.000002196`
    - `stddev_rms_output_abs_diff=0.000010570`

### Decision Impact

- `FP16 + stochastic rounding` is now better supported than before:
  - not only on synthetic traffic and one-step drift
  - but also on accumulated layer-derived chain drift with multiple seeds
- the new result strengthens the current GB10 policy:
  - plain `fp16` remains the speed floor
  - `fp16_sr` remains the better production candidate because its accumulated drift is much lower and its seed-to-seed spread is small

### Remaining Risk

- the chain still reuses the same deterministic decode operands at each step; it is not yet a full real-token decode stream
- the next expansion should widen step counts or introduce a more realistic decode operand evolution model, not replace the current fixture-chain baseline
- the next realistic expansion should follow the actual target workload from the main plan:
  - long shared system root
  - append-only multi-turn chat
  - committed-head restore plus turn-tail append

## 2026-03-29 - Target-Shaped Mamba Trace Replay

### Goal

Add a realism-oriented Mamba drift benchmark that matches the actual serving profile better than the earlier fixed-operand chain.

### Scope

- add a `fixture_trace` operation to the GB10 Mamba benchmark
- generate a target-shaped layer-0 trace fixture with:
  - shared system root on the order of `8k` tokens
  - committed conversation head at `49,152` tokens
  - append-only multi-turn user-tail prefill and assistant decode phases
- support `fp32`, `fp16`, and `fp16_sr`
- keep multi-seed drift statistics for `fp16_sr`

### Key Code And Artifacts

- [benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu)
- [tools/oracle/dump_mamba_target_trace_fixture.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/dump_mamba_target_trace_fixture.py)
- [tools/oracle/generate_mamba_target_trace_fixture.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/generate_mamba_target_trace_fixture.sh)
- [testing/oracle/mamba_layer0_target_chat_trace/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace/metadata.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T045219Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T045219Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T045219Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T045219Z_cuda132.env.txt)

### Result

- the benchmark now has six operation classes:
  - `clone`
  - `update`
  - `mamba_update`
  - `fixture_update`
  - `fixture_chain`
  - `fixture_trace`
- the new target-shaped fixture replays `704` total steps across realistic phase boundaries:
  - `192` user-tail prefill steps
  - `128` assistant decode steps
  - `128` user-tail prefill steps
  - `96` assistant decode steps
  - `96` user-tail prefill steps
  - `64` assistant decode steps
- at `704` steps:
  - `1` request:
    - `fp16 mean_output_abs_diff=0.012537245`
    - `fp16_sr mean_output_abs_diff=0.000258212`
    - `fp16_sr stddev_mean_output_abs_diff=0.000004984`
  - `8` requests:
    - `fp16 mean_output_abs_diff=0.012537245`
    - `fp16_sr mean_output_abs_diff=0.000262101`
    - `fp16_sr stddev_mean_output_abs_diff=0.000001901`

### Decision Impact

- the Mamba cache-format decision is now supported by a more realistic serving-profile benchmark, not only by fixed-operand chain replay
- `FP16 + stochastic rounding` remains the lead optimized-cache candidate:
  - it stays much closer to FP32 than plain `fp16`
  - its seed-to-seed spread remains small on the target-shaped replay

### Remaining Risk

- the trace fixture still uses deterministic synthetic per-step operands rather than a full reference-backend token trace
- the current benchmark reports whole-trace drift; it does not yet split drift growth by phase boundary
- the next expansion should either:
  - widen the realistic trace lengths further toward the service envelope
  - or derive the trace from a richer oracle source

## 2026-03-29 - Target-Trace Phase-End Checkpoints

### Goal

Make the realistic `fixture_trace` replay easier to interpret by automatically capturing the user-tail and assistant-decode boundaries that matter for the target serving pattern.

### Scope

- parse `trace_phases` from the target-trace metadata
- auto-include phase-end checkpoints in `fixture_trace`
- label result rows with the matching phase name and kind
- emit phase-end checkpoint metadata into the JSON artifact

### Key Code And Artifacts

- [benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench.cu)
- [testing/oracle/mamba_layer0_target_chat_trace/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace/metadata.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T050137Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T050137Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T050137Z_cuda132.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T050137Z_cuda132.env.txt)

### Result

- the realistic trace now runs at:
  - requested checkpoints: `128, 320, 544, 704`
  - auto-added phase ends: `192, 448, 640`
- the JSON artifact now records:
  - `fixture_trace_phase_end_steps`
  - `trace_phase_name`
  - `trace_phase_kind`
- the phase-end results show where drift grows across the real serving shape:
  - `1` request, `192` steps (`user_turn_1_tail_prefill`)
    - `fp16 mean_output_abs_diff=0.003934261`
    - `fp16_sr mean_output_abs_diff=0.000162880`
  - `1` request, `640` steps (`user_turn_3_tail_prefill`)
    - `fp16 mean_output_abs_diff=0.009664417`
    - `fp16_sr mean_output_abs_diff=0.000189491`
  - `1` request, `704` steps (`assistant_turn_3_decode`)
    - `fp16 mean_output_abs_diff=0.012537245`
    - `fp16_sr mean_output_abs_diff=0.000258212`

### Decision Impact

- the realistic Mamba replay is now easier to reason about in service terms because drift can be read at turn-tail and assistant-decode boundaries
- `FP16 + stochastic rounding` remains the lead optimized-cache candidate not only at whole-trace end, but across the intermediate phase boundaries that matter for multi-turn chat

### Remaining Risk

- phase-end checkpoints are a good first cut, but they still do not expose intra-phase drift growth
- the trace source is still deterministic synthetic replay rather than a full reference-backend token trace

## 2026-03-29 - Oracle-Derived Mamba Trace v2 And Comparison

### Goal

Replace the older synthetic target-trace inputs with a richer oracle-derived multi-turn profile, then turn the result into an explicit synthetic-vs-oracle comparison artifact.

### Scope

- add a structured multi-turn chat profile for target-trace dumping
- derive the trace from real tokenization and real layer-0 projected operands
- record actual tokenized counts alongside the service-profile target counts
- add a comparison tool that lines up synthetic and oracle-derived phase rows by phase identity instead of raw step count

### Key Code And Artifacts

- [testing/oracle/mamba_target_chat_profile.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_target_chat_profile.json)
- [tools/oracle/dump_mamba_target_trace_fixture.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/dump_mamba_target_trace_fixture.py)
- [tools/oracle/generate_mamba_target_trace_fixture.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/generate_mamba_target_trace_fixture.sh)
- [testing/oracle/mamba_layer0_target_chat_trace/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace/metadata.json)
- [tools/benchmark_analysis/compare_mamba_trace_benchmarks.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/benchmark_analysis/compare_mamba_trace_benchmarks.py)
- [benchmarks/gb10_mamba_trace_compare/run_compare.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_trace_compare/run_compare.sh)
- [artifacts/benchmarks/gb10_mamba_cache_20260329T052927Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T052927Z_cuda132.json)
- [artifacts/benchmarks/gb10_mamba_trace_compare_20260329T053747Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_compare_20260329T053747Z.md)

### Result

- the current oracle-derived target trace now records both service targets and actual tokenized counts:
  - service targets: `8192` shared-root, `49152` committed-head
  - actual counts: `1436` shared-root, `2820` committed-head
- the dumped trace replays `1566` total steps across six labeled phases:
  - `323` user-tail
  - `304` assistant decode
  - `277` user-tail
  - `261` assistant decode
  - `203` user-tail
  - `198` assistant decode
- the new comparison report shows that the richer trace changes the interpretation of the Mamba format results:
  - plain `fp16` mean drift is lower than on the synthetic phase-end control at all matched phase rows
  - plain `fp16` max drift is higher than on the synthetic phase-end control at all matched phase rows
  - `fp16_sr` mean drift and variance are both materially higher than on the synthetic phase-end control
  - `fp16_sr` still beats plain `fp16` on mean drift, but the margin shrinks from roughly `22x-51x` to roughly `1.1x-3.4x`

### Decision Impact

- `FP16 + stochastic rounding` remains the leading optimized Mamba cache-format candidate
- but the current policy should be based on:
  - lower mean drift than plain `fp16`
  - bounded variance
  - acceptance that the safety margin is smaller on realistic data patterns than the synthetic trace suggested
- the richer trace is now the realism anchor, while the synthetic trace remains the control

### Remaining Risk

- the current oracle-derived trace is still a scaled proxy rather than a full service-scale `8k/49k` trace
- one oracle-derived conversation profile is not enough to freeze the production threshold
- the next realism step should be additional oracle-derived profiles, not removal of the synthetic control

## 2026-03-29 - Second Oracle-Derived Structured Trace

### Goal

Add a second oracle-derived multi-turn profile with different serialization and token-distribution characteristics, then compare it directly against the first oracle-derived trace.

### Scope

- add a second structured profile with the same phase names and kinds
- dump a second layer-0 oracle-derived trace fixture from that profile
- run the existing `fixture_trace` benchmark against the alternate trace root
- compare the two oracle-derived artifacts phase by phase

### Key Code And Artifacts

- [mamba_target_chat_profile_structured.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_target_chat_profile_structured.json)
- [mamba_layer0_target_chat_trace_structured/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace_structured/metadata.json)
- [gb10_mamba_cache_20260329T054243Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T054243Z_cuda132.json)
- [gb10_mamba_trace_compare_oracle_20260329T054506Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_compare_oracle_20260329T054506Z.md)

### Result

- the new structured oracle-derived profile tokenizes to:
  - `1987` shared-root tokens
  - `3555` committed-head tokens
  - `2012` replayed steps
- on the structured profile itself:
  - `fp16_sr` still beats plain `fp16` on mean drift at every phase boundary
  - the mean-drift advantage ranges from roughly `1.3x` to `4.1x`
- the oracle-vs-oracle comparison shows:
  - plain `fp16` mean drift is higher on all matched rows for the structured profile
  - `fp16_sr` mean drift is higher on `8/12` matched rows and max drift is higher on `4/12`

### Decision Impact

- the optimized candidate is now stable across two oracle-derived traces:
  - `FP16 + stochastic rounding` still beats plain `fp16` on mean drift
- but the exact margin is profile-sensitive, so the policy should be based on a small family of oracle-derived traces rather than on a single realism anchor

### Remaining Risk

- two oracle-derived profiles are better than one, but still not enough to freeze a production threshold
- the next realism step should be a third profile or a longer-profile variant, not a broader synthetic sweep

## 2026-03-29 - Third Oracle Profile And Family Summary

### Goal

Add a third oracle-derived trace with a tool/schema-oriented serialization shape and replace ad hoc pairwise reading with a family-summary artifact.

### Scope

- fix the trace generator so literal braces are supported in prompt text
- add a third tool-style oracle-derived profile
- benchmark that profile through `fixture_trace`
- summarize the current oracle family in one artifact

### Key Code And Artifacts

- [dump_mamba_target_trace_fixture.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/dump_mamba_target_trace_fixture.py)
- [mamba_target_chat_profile_toolish.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_target_chat_profile_toolish.json)
- [mamba_layer0_target_chat_trace_toolish/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace_toolish/metadata.json)
- [summarize_mamba_trace_family.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/benchmark_analysis/summarize_mamba_trace_family.py)
- [gb10_mamba_cache_20260329T144225Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T144225Z_cuda132.json)
- [gb10_mamba_trace_family_20260329T144543Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_family_20260329T144543Z.md)

### Result

- the generator now supports literal braces in profile text without awkward escaping
- the new tool-style oracle-derived profile tokenizes to:
  - `1712` shared-root tokens
  - `3330` committed-head tokens
  - `1798` replayed steps
- the family summary now shows all three oracle-derived traces together

### Decision Impact

- the optimized candidate is now supported by a family-level read:
  - `fp16_sr` still beats plain `fp16` on mean drift at every measured phase boundary
- the weakest current family range is now explicit:
  - early user-tail phase: about `1.06x-1.31x`
- the strongest current family range is also explicit:
  - late user-tail / assistant phases: about `3.1x-4.1x`
- that means threshold-setting should anchor on the early user-tail phases, not on the best-case later phases

### Remaining Risk

- three oracle-derived traces are better than two, but still not enough to freeze a production threshold
- the next realism step should keep expanding the oracle family and tracking the weakest-margin phases

## 2026-03-29 - Fourth Oracle Profile And Guardrail Anchors

### Goal

Add a fourth oracle-derived target trace with a markdown-heavy serialization pattern and turn the family summary into a direct observed-guardrail report.

### Scope

- add a fourth markdown-heavy target-chat profile
- generate and benchmark its oracle-derived trace fixture
- extend the family-summary tool with observed floor/ceiling output
- verify any surprising result with a targeted rerun before updating policy

### Key Code And Artifacts

- [mamba_target_chat_profile_markdownish.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_target_chat_profile_markdownish.json)
- [mamba_layer0_target_chat_trace_markdownish/metadata.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace_markdownish/metadata.json)
- [summarize_mamba_trace_family.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/benchmark_analysis/summarize_mamba_trace_family.py)
- [run_guardrails.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/gb10_mamba_trace_family/run_guardrails.sh)
- [gb10_mamba_cache_20260329T145158Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T145158Z_cuda132.json)
- [gb10_mamba_cache_20260329T145444Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_cache_20260329T145444Z_cuda132.json)
- [gb10_mamba_trace_family_20260329T145706Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_family_20260329T145706Z.md)
- [gb10_mamba_trace_guardrails_20260329T145706Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_guardrails_20260329T145706Z.md)

### Result

- the new markdown-heavy profile tokenizes to:
  - `1655` shared-root tokens
  - `3503` committed-head tokens
  - `2255` replayed steps
- the family summary now includes four oracle-derived traces
- the new guardrail report records observed per-phase:
  - weakest `fp16/fp16_sr` ratio
  - `fp16_sr` mean-drift ceiling
  - `fp16_sr` max-spike ceiling
  - `fp16_sr` mean-stddev ceiling

### Decision Impact

- the stronger family view changed the policy:
  - `fp16_sr` is still the strongest overall optimized candidate
  - but it no longer beats plain `fp16` on every measured oracle phase
- the weakest verified point is now:
  - `1` request, first `user_turn_1_tail_prefill`: `fp16/fp16_sr = 0.925123`
- the `8`-request version of the same phase is near parity:
  - `fp16/fp16_sr = 1.019608`
- this blocks production guardrail freeze and shifts the next step toward understanding the early-phase regression

### Remaining Risk

- one verified regression at the weakest phase is enough to invalidate the old “uniformly better” reading
- the next realism step should be:
  - one more profile or longer variant
  - plus per-phase operand-range or activation-shape analysis to explain the markdown-heavy regression

## 2026-03-29 - First-User-Tail Weakness Investigation And Mamba Format Decision

### Goal

Understand and resolve the `fp16_sr` regression in the first user-tail phase, then reach a production format decision.

### Scope

- operand analysis, token hotspot mapping, and dt-bucket analysis across the oracle family
- controlled serializer variants (flat, headerless) to isolate the cause
- think-boundary analysis to rule out `<think>` as a differential driver
- dt-scale sweep to test the causal role of dt amplitude
- `--dt-threshold` gating experiment (SR→RN when `|dt| > 0.65`)

### Key Findings

1. The regression correlates with the `dt` operand-distribution: profiles spending more time in high-dt regimes show worse `fp16/fp16_sr` ratios (Pearson `r = -0.86` for ratio vs share above p75 dt)
2. No single token pattern is the root cause: removing bullets (flat variant), headings (headerless variant), or think-tags all failed to eliminate the weakness
3. The dt-scale sweep shows a **non-monotonic** response — 1.25x dt actually *helps* (`ratio@1 = 1.24`) while 1.0x and 1.5x are worse — so simple dt-amplitude explanations fail
4. Global `|dt| > 0.65` threshold gating improves the weak phase (`0.89 → 0.94` @1req) but causes unacceptable later-phase regressions (worst delta `-0.71`)
5. Think-boundary tokens are invariant hotspots across all profiles, not a differential driver

### Key Artifacts

- dt-bucket analysis: [gb10_mamba_trace_dt_buckets_20260329T163241Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_dt_buckets_20260329T163241Z.md)
- dt-scale sweep: [gb10_mamba_trace_dt_scale_sweep_20260329T164125Z_cuda132.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_dt_scale_sweep_20260329T164125Z_cuda132.md)
- Threshold gating rejection: [gb10_mamba_dt_threshold_0p65_family_20260329T165952Z_cuda132.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_dt_threshold_0p65_family_20260329T165952Z_cuda132.md)
- Latest guardrails: [gb10_mamba_trace_guardrails_20260329T152925Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_guardrails_20260329T152925Z.md)
- Latest operand report: [gb10_mamba_trace_operands_20260329T152925Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_operands_20260329T152925Z.md)

### Decision

Accept `FP16 + stochastic rounding` as the production format with the known first-user-tail weakness:

- the weakness is a complex operand-distribution interaction, not a correctable single-parameter defect
- even at the worst observed point (`fp16/fp16_sr = 0.848`), both formats produce mean drift under `0.001`
- `fp16_sr` wins convincingly on all later phases (`1.5x-4.1x` advantage)
- re-assess when production traces are available
