# Nemotron Inference Runtime

From-scratch C++/CUDA inference engine for Nemotron-3 hybrid Mamba-attention-MoE models on RTX 5090.

## Development Workflow

When starting a substantial new development focus (new kernels, new subsystems, multi-file changes, performance optimization), use `/plan-init` → `/implement`:

1. `/plan-init <description>` — creates `proj-YYYY-MM-DD-HHMM/PLAN.md` with context, references, rules, and steps
2. Review the plan with the user; send to Codex for factual review (`/cx-delegate --background --fresh`)
3. Commit the reviewed plan before implementation starts
4. `/implement <project-dir>/PLAN.md` — executes all steps: delegate → review → fix → test → commit → next

Shared rules and the test protocol live in `PLAN_RULES.md` at the project root. Both commands read it.

### Key principles

- **The plan is ground truth.** Implementation follows the plan's intent.
- **One commit per step.** Clean git history maps 1:1 to plan steps.
- **Diagnose before fixing.** When something doesn't work, add a diagnostic step first. Don't guess at fixes. See "Probe-then-decide" below for the tooling.
- **Run on hardware.** Compile-only verification is necessary but not sufficient. Benchmarks reveal real bottlenecks.
- **Update the plan with findings.** When diagnosis reveals a different problem, rewrite the steps — don't implement the wrong fix.
- **Reference implementations are guardrails, not blueprints.** Align with the reference algorithm but justify divergences explicitly.
- **Always pass the analysis notes file** (`/home/khkramer/.claude/plans/*.md`) to Codex delegations so it has full context.
- **Stream benchmark output.** Use `stdbuf -oL` and `std::unitbuf` so long runs can be monitored.

### Probe-then-decide

For kernel work that touches CUTE/CUTLASS layouts, MMA atom contracts, TMA descriptors, or any compile-time CUTLASS template instantiation: **probe the actual values before writing a fix.** Speculating about what a layout does when the compiler can answer in seconds wastes days. Tools, ordered by cost — use the cheapest one that can falsify the current hypothesis:

1. **Compile-time layout probes** (seconds). Extract any constexpr value from a static CUTE `Layout` or `TiledMma` via the incomplete-type trick:
   ```cpp
   template <int V> struct ShowInt;  // intentionally undefined
   [[maybe_unused]] ShowInt<static_cast<int>(cute::cosize(TheLayout{}))> probe;
   ```
   Build the file; nvcc prints `ShowInt<N>` in the diagnostic with the actual integer. Works for `cute::size`, `cute::cosize`, `cute::tile_size<i>`, `cute::size<i>` on anything fully static. Revert the probe before committing.

2. **Synthetic-data oracle tests** (minutes to write). Construct an exact FP32 reference and compare the kernel's output **bitwise** — not "within tolerance". Templates in tree:
   - `testing/backend/p13_generic_direct_stage_oracle_test.cpp` — full mainloop+epilogue oracle with `nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited` accumulator seeding from a live `partition_C` coord map.
   - `testing/backend/p1_natural_fp4_mma_oracle_test.cpp` — mainloop sentinel against a TiledMma's natural orientation.
   - `testing/backend/swap_false_fp4_oracle_test.cpp` — host-only FP4 dequant + fp64 GEMM reference for ground-truth construction.

3. **Fragment-level dump oracles** (hours). When an end-to-end oracle diverges but layout coverage is provably correct, bisect the kernel's data pipeline by dumping per-lane register state at each stage (post-`LoadFragment*`, post-`cute::fp4_shift_A/B`, post-`mma_atom.call`, post-`partition_C` writeback) and comparing each to a host model of that stage. Frozen as a sentinel after the bug is found.

4. **Runtime layout dump CUs** under `artifacts/tmp/trt_*_runtime_layout_dump.cu`. Small standalone CUs that build and print `partA / partB / partC`, the scale TV layouts, and `accum_profile` layouts at runtime. Use when characterizing a new `TiledMma` / `CollectiveMainloop` / `MmaTileShape` instantiation. Each run produces a `.log` artifact under `artifacts/benchmarks/` that can be checked into the source as an authoritative comment block alongside the type alias.

**Order of operations when a kernel is wrong:** never write a kernel fix before the cheapest probe that can answer the open question has been run. Never write an end-to-end integration test as the *first* validation of a new layout — write the compile-time probe first, then a synthetic-data oracle, then the integration test.

**Verify the test reference before trusting a bisection result.** When a fragment-level oracle reports a failing bucket, the very first follow-up probe should compare the test's input staging against its own host-side reference, not against the kernel's behavior. The kernel may be correct and the test wrong. The cheap pattern: dump the staged smem bytes by physical offset cooperatively from the device, then bytewise compare against a host computation of the same physical layout. If they match, the bug is in the kernel's load path; if they diverge, the bug is in the test's reference computation or its staging code. This pattern has fired twice in the P1 fragment-debug bring-up (first on `tCrSFA`, then on `tCrA_pre_shift`); both initial diagnoses were test artifacts, not kernel bugs.

**Tolerance gates** (`max_abs_diff <= ε`) belong in *integration* tests against production. Kernel-level correctness for native FP4 MMA paths must be verified **bitwise** against an FP32 reference computed from the same dequantized inputs. BF16-noise tolerance hides MMA-level bugs and has burned us before — see the P1-FC1 bring-up that produced this section.
