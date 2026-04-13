# Shared plan rules

Include these rules in every `PLAN.md` under the `## Rules` section, in addition to any plan-specific rules.

## Kernel Provenance (hard constraint)

### The rule

On the **prefill/decode hot path** (any code that actually runs during a production inference request), only **from-scratch custom CUDA kernels** are permitted. No CUTLASS templates, collectives, or builders. No FlashInfer, TRT-LLM, or vLLM runtime code.

### The rationale

The goal is not just ownership — it is a performance theory: a kernel hand-tuned for exactly one model (Nemotron-3 Super NVFP4), one hardware target (**RTX 5090 / SM120**), and one set of shapes can beat vendor library kernels that have to generalize. SM120 NVFP4 MoE support is still immature in upstream libraries, so there is a real specialization edge. The rule captures that edge at the narrowest possible scope. Plan authors are told to assume the performance theory holds rather than relax the rule under performance pressure.

### What is explicitly permitted on the hot path

- Our own CUDA code, written from scratch.
- **SM120 block-scaled MMA via inline PTX / intrinsics.** Hardware MMA is hardware, not library code.
- **CUTE atoms and layout primitives** — specifically: `cute::MMA_Atom<cute::SM120_16x8x64_TN_VS<...>>`, `cute::TiledMMA`, `cute::Copy_Atom<cute::SM90_TMA_LOAD>`, `cute::Copy_Atom<cute::SM75_U32x4_LDSM_N>`, `cute::Layout`, `cute::Tensor`, swizzle functors, and `cutlass::detail::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB`. These are templated layout/tile primitives, not kernel machinery.
- **Study of reference stacks** (CUTLASS examples, vLLM, TRT-LLM, FlashInfer, generated PTX) as guidance that then gets **re-emitted as hand-written CUDA** the runtime fully owns.

### What is explicitly forbidden on the hot path

- `cutlass::gemm::device::GemmUniversal`, `cutlass::gemm::device::GemmUniversalAdapter`, any `GemmKernel`
- `cutlass::gemm::collective::CollectiveBuilder`, `cutlass::epilogue::collective::CollectiveBuilder`
- `KernelScheduleAuto`, `EpilogueScheduleAuto`, `TmaWarpSpecialized*` schedule tags
- `CollectiveMma`, `CollectiveEpilogue`
- Any `MainloopSm*` dispatch-policy type
- Any header under `cutlass/gemm/collective/`, `cutlass/gemm/device/`, `cutlass/gemm/kernel/`, or `cutlass/epilogue/collective/`
- FlashInfer, TRT-LLM, or vLLM runtime code

### Transitional surfaces — leave existing working code alone

Existing, working, tested hot-path code that uses forbidden primitives is **grandfathered**: **do not refactor it just to enforce this policy.** It stays live until a from-scratch replacement for the same compute stage is ready to promote to default dispatch, at which point the transitional surface is retired in the same commit. Temporary means temporary; a transitional surface that stays past its replacement is a rule violation.

**When you do touch transitional code for an unrelated reason, follow the Kernel Provenance rules on the touched surface** — new kernels you add or existing kernels you meaningfully rewrite must be from-scratch. Mechanical renames, comment cleanup, and bug fixes that do not change the kernel structure may stay on the existing CUTLASS-based path; meaningful algorithmic changes trigger the rewrite requirement.

Current transitional surfaces in this repo (non-exhaustive — grep `cutlass::gemm::collective::CollectiveBuilder` under `runtime/` for the authoritative list):
- `TracedP5CollectiveMainloop` / `TracedP5*` bundle in `runtime/src/backend/fused_moe_prefill.cu`
- `TracedP13CollectiveMainloop` / `TracedP13*` bundle (same file)
- `TracedP15CollectiveMainloop` / `TracedP15*` bundle (same file)

Any NEW hot-path kernel work MUST be from-scratch per this policy. Adding new `CollectiveBuilder`-based types to `runtime/` is a policy violation even if they follow the TracedP* pattern, unless those types are explicitly replacing a transitional surface at promotion time.

### Off the hot path — unrestricted

- Tooling, probes, tests, benchmarks, diagnostic scripts, oracle generators, one-off experiments can freely use any library (CUTLASS collectives, FlashInfer, TRT-LLM, vLLM, cuBLASLt, cuDNN, PyTorch, pandas). The rule applies only to production-request code.
- Reference/oracle benchmarks and tests may compare against library backends as a special case of the unrestricted-tooling clause. Step 2 of `proj-2026-04-12-1022/PLAN.md` (the flashinfer BF16 `gemm1_output` capture harness) is an example: it patches and drives a library backend to generate a reference artifact, but the library code never runs inside Nemotron's own runtime.

## Safety and sequencing
- Do not disable a working code path until its replacement is fully validated and passing the complete test suite plus prompt-length sweep. Keep the existing path as a working fallback throughout development. (Lesson: proj-2026-04-10-2021 step 8 removed the BF16 buffer before having a validated FP32 replacement, which disabled the only working grouped FC1 path and caused inference divergence.)
- Reuse existing payload buffers unless duplication is required by a measured optimization. Do not allocate new buffers just to mirror another path.
- New kernels and launcher paths should enter the existing planning, tracing, and benchmarking architecture (`AppendLinearOpTraceEntry`, `GemmKernelFamily`, `GemmHeuristicCache`). They should not be untracked one-off launcher paths.
- Kernels must support arbitrary token counts (1-512+), including partial row tiles.

## Validation categories
Ordered from highest signal to lowest:
- **Per-layer diagnostic** (`NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`): compares the prefill path against the single-token decode path per layer and per token. This is the most sensitive detector of per-layer drift.
- **Behavioral parity oracle on RTX 5090**: constrained greedy comparison against local vendored `vllm` via `tools/oracle/compare_chat_runtimes.py`. This is the ground truth for whether the runtime produces correct output.
- **Secondary external reference**: local vendored `trtllm`, useful for gross regression detection but not an exact implementation oracle.
- **Claude coherence scoring**: smoke test only, not acceptance proof.
- **Primary performance gate**: end-to-end prefill latency.
- **Secondary diagnostic metrics**: hot kernel time and cold descriptor-build/setup overhead. These metrics exist to explain end-to-end results, not replace them.

## Resource constraints
- Do not increase per-CTA shared memory allocation beyond what the existing kernel uses. Reuse mainloop shared memory for epilogue work after the last K-tile iteration.
- GPU memory is tight (RTX 5090, 32GB, fully consumed by the model). Workspace allocations must be justified and measured.

## Test protocol

After each implementation step, run the tests appropriate for the type of code changed. All GPU tests must run **sequentially** (`-j1`) — the RTX 5090's 32GB is fully consumed by the model, and parallel GPU tests will OOM.

**Known pre-existing failures** (do not count as regressions):
- `nvfp4_weight_test` — model weight loading
- `expert_layer_oracle_test` — model weight loading
- `expert_layer8_oracle_test` — model weight loading

### After any code change
```
cmake --build build --parallel $(nproc)
```

### After runtime/backend code changes (C++/CUDA in `runtime/src/`)
```
ctest --test-dir build --output-on-failure -j1
```
Expected: 68/71 pass (3 known failures above). Any new failure is a regression — diagnose before proceeding.

### After kernel changes affecting MoE prefill
Also run the focused regression oracle:
```
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build/testing/nano_24_token_prefill_regression_test
```
This must produce exact token-oracle matches for both prompt23 and prompt24.

### After changes that could affect inference quality
Also run the prompt-length sweep:
```
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 uv run tools/oracle/prompt_length_sweep.py --runtimes native --skip-claude-eval
```
Expected: `ALL CHECKS PASSED`. Any degenerate length is a regression.

### For validation/benchmark steps
Run the full validation hierarchy from the "Validation categories" section above, including vLLM parity and per-layer diagnostic.
