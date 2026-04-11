# Shared plan rules

Include these rules in every `PLAN.md` under the `## Rules` section, in addition to any plan-specific rules.

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
