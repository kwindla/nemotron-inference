# Preprocessor gate cleanup + `fused_moe_prefill.cu` umbrella split

## Context & motivation

Session `86e37198-c2d2-460f-8096-61c13808afb6` ran for 8.5 hours on 2026-04-13 attempting to fix the Phase 3 failure in `nano_p1_mainloop_oracle_test` (Nano K=2688 bucket, 230,087 mismatches / 93.6% before the session, same afterward). The session compacted 16 times and produced zero commits. Post-mortem analysis (see "Investigation Facts" below) identified two root causes that made the session unable to converge:

1. **Context thrash from `fused_moe_prefill.cu`.** The file is 13,716 lines / 547 KB (~137k tokens). It is the primary target of the NanoP1 debugging work and gets automatically re-attached by Claude Code on every compaction via `compact_file_reference`. At a typical compaction budget of ~150k context tokens, re-attaching the file leaves single-digit-thousand tokens for productive work — so every compaction cycle produces ~20-30 minutes of useful work before the next compaction. The session completed 16 cycles of this pattern, rediscovering the same facts each time.

2. **Preprocessor gates constrain any split.** The file is heavily littered with `#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)` gates (63 occurrences), plus dead `NEMOTRON_P5_PARTITION_DEBUG` / `NEMOTRON_P5_LINEAR_PARTITION_DEBUG` blocks that are never defined, plus `__CUDA_ARCH__ >= 900` / `>= 1000` gates. One of the `LOCAL_CUTE` blocks spans 1,548 lines (7058→8606); another contains the entire NanoP1 kernel territory (12189→12847, 658 lines). A naive extraction into smaller files fails because cuts that cross `#if`/`#endif` boundaries sever the gates.

The cleanup has two phases:
- **Delete all preprocessor gates** (the codebase targets a single model, single hardware, single compiler; gates are always dead scaffolding for alternate configurations that don't exist).
- **Split `fused_moe_prefill.cu` into an umbrella `.cu` file that `#include`s multiple `.cuh` fragments.** Each fragment is scoped to one kernel family, ~100-800 lines each. Claude Code's compaction reload then attaches the edited fragment rather than the entire 13k-line file, dropping auto-reload cost from ~147k → ~20k tokens per compaction cycle.

The NanoP1 Phase 3 bug is **not** fixed by this plan. This plan restores the ability for a new session to work on the fix without immediately running into context thrash.

## Goal

1. Remove every preprocessor gate from the nemotron-inference codebase that is not an include guard. No `#if` / `#ifdef` / `#ifndef` blocks for config, debug, platform, architecture, or compiler detection should remain.
2. Refactor `runtime/src/backend/fused_moe_prefill.cu` (13,716 lines) into a thin umbrella `.cu` file (~150-200 lines) that `#include`s multiple `.cuh` fragments, one per kernel family.
3. Rewrite active-plan line-number references to `fused_moe_prefill.cu:<n>` into symbol-name references that survive future line renumbering.
4. Leave all kernel logic, all test expectations, and all binary behavior unchanged. The Phase 3 failure count must be identical before and after this plan.

## Rules

**Read `PLAN_RULES.md` at repo root for shared rules** (kernel provenance, safety, test protocol, resource constraints, validation categories, pre-existing failures). This plan does not override any of those rules.

Plan-specific rules:

1. **Zero logic changes.** Every commit in this plan is a mechanical text transform: gate deletion, file splitting, or reference rewriting. No algorithmic changes, no bug fixes, no optimizations.

2. **Transitional surfaces may be touched under the cleanup exemption.** Per `PLAN_RULES.md`, "Mechanical renames, comment cleanup, and bug fixes that do not change the kernel structure may stay on the existing CUTLASS-based path." Preprocessor gate deletion and file splitting are mechanical cleanup; they do not change kernel structure. The `TracedP5CollectiveMainloop`, `TracedP13CollectiveMainloop`, and `TracedP15CollectiveMainloop` bundles in `fused_moe_prefill.cu` are grandfathered transitional surfaces — they will be touched for gate removal and fragment extraction but their CUTLASS-based implementation is untouched.

3. **Verify binary equivalence, not just "it compiles".** Each commit must pass:
   - Clean `cmake --build build -j`.
   - Full `ctest -j1` with the same pass/fail count as before the commit (68/71 pass expected, 3 known failures per `PLAN_RULES.md`).
   - `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build/testing/nano_24_token_prefill_regression_test` with exact token-oracle matches for prompt23 and prompt24, per `PLAN_RULES.md`.
   - `nano_p1_mainloop_oracle_test` Phase 3 mismatch count must be identical to the pre-commit baseline (230,087 at the time of plan authoring; re-baseline at the start of execution since the killed session had WIP in the tree).
   - For Step 0 specifically: `nvcc -E` preprocessed output diff for the affected CUDA translation units must contain only lines from deleted dead branches — no other differences.

4. **One step per commit. One commit per step.** Clean git history where every commit is revertible independently.

5. **No gate is sacred.** If an `#if` block exists in the code after Step 0 and it is not an include guard, it is a bug. Includes `__CUDA_ARCH__`, `CUDA_VERSION`, `__CUDACC__`, `_WIN32`, `__linux__`, `NEMOTRON_*` — all of them go. The rationale is captured in this plan's Context & motivation section: the target is a single model / single hardware / single compiler environment, and the remaining gates are dead scaffolding that also blocks the file split.

## Investigation facts

The following are verified against the current working tree at the time of plan authoring. Codex should re-verify each before execution, since the tree state may shift.

### Session forensics

- Target session: `86e37198-c2d2-460f-8096-61c13808afb6` (jsonl at `~/.claude/projects/-home-khkramer-src-nemotron-inference/`).
- Duration: 2026-04-13 16:17 UTC → 00:54 UTC = 8h 37m.
- Compactions: 16.
- Commits produced: 0.
- Phase 3 mismatch trajectory: 230,087 → 243,611 (worse after a speculative fix) → 230,087 (back to original). No net progress.
- Hypothesis churn: 3 distinct root-cause theories across the session, each rediscovered after a compaction cleared prior context.
- Auto-reloaded files per compaction (`compact_file_reference` attachments): current re-count from the session jsonl is `fused_moe_prefill.cu` 16/16, `nano_p1_mainloop_oracle_test.cpp` 14/16, `capture_bf16_gemm1.py` 6/16, `proj-2026-04-12-1022/PLAN.md` 5/16, `NOTES.md` 2/16, plus a handful of one-off library/reference files. Treat these counts as forensics, not a contract.

### File sizes and auto-reload cost

| File | Size | Lines | Approx. tokens | Reloads / 16 |
|---|---:|---:|---:|---:|
| `runtime/src/backend/fused_moe_prefill.cu` | 547 KB | 13,716 | ~137k | 16 |
| `testing/backend/nano_p1_mainloop_oracle_test.cpp` | 38 KB | 1,114 | ~9.6k | 14 |
| `proj-2026-04-12-1022/PLAN.md` | 62 KB | 499 | ~15.6k | 5 |
| `proj-2026-04-12-1022/trtllm_reference/NOTES.md` | 33 KB | 377 | ~8.4k | 2 |
| `testing/backend/nano_p1_direct_pack_oracle_test.cpp` | 36 KB | 987 | ~9.1k | 1 |
| `proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py` | 16 KB | — | ~4.1k | 6 |

Worst-case single-compaction reload cost: ~180k tokens (compaction #1). Typical: ~147k. Post-split projected best case: ~20k, but Step 3 explicitly validates whether Claude Code actually follows fragments rather than the umbrella TU.

### Preprocessor gate inventory

Verified via `grep -rE '^\s*#(if|ifdef|ifndef)\b'`:

#### `runtime/src/backend/fused_moe_prefill.cu` — 63 gate lines

- **`NEMOTRON_P5_PARTITION_DEBUG` blocks — all dead code, never defined anywhere in repo:**
  - line 1892 → 1895 (3 lines of body)
  - line 1979 → 1981 (2 lines)
  - line 5599 → 5636 (37 lines)
  - line 8491 → 8513 (22 lines)
- **`NEMOTRON_P5_LINEAR_PARTITION_DEBUG` blocks — all dead code, never defined anywhere:**
  - line 1897 → 1900 (3 lines)
  - line 1983 → 1985 (2 lines)
  - line 5638 → 5661 (23 lines)
  - line 8515 → 8542 (27 lines)
- **`NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE` blocks — macro is unconditionally defined wherever the file is compiled (see "CMake macro sources" below), so `#if` branches are live, `#else` branches are dead:**
  - lines with `#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)`: 25, 44, 62, 107, 127, 692, 703, 1750, 3188, 4552, 4650, 4719, 4776, 4844, 4888, 4932, 4973, 5028, 5087, 5156, 5230, 5291, 5430, 5484, 5543, 5685, 7058, 8634, 8705, 8718, 8732, 8745, 9082, 9099, 9119, 9136, 9153, 9158, 9184, 9324, 9392, 9405, 9419, 9432, 9473, 11151, 12189, 12998, 13041, 13074, 13209 (~51 opening gates)
  - Paired closings (`#endif`, often with `#else` in between) at corresponding locations.
  - Largest single block: line 7058 → 8606 = **1,548 lines**, contains `#else` at 8582.
  - Contains the NanoP1 kernel: line 12189 → 12847 = 658 lines (no `#else`).
- **`__CUDA_ARCH__ >= 900` / `>= 1000` gates — SM120 satisfies both, so `#if` branches live, fallbacks dead:**
  - lines 3771 → 3773
  - lines 3892 → 3894
  - lines 4201 → 4226

#### `runtime/src/backend/device_nvfp4_matrix.cu` — 7 gate lines

All `#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)` at lines 18, 852, 926, 1012, 1086, 1172, 1245. All in scope: delete the `#else` branches (if any), inline the `#if` branches.

#### `runtime/include/nemotron/p15_scale_helpers.h` — 1 gate

`#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)` at line 8, `#endif` at line 679. Wraps the entire file. Delete the wrapper; no conditional content.

#### `runtime/include/nemotron/p15_scale_runtime_helpers.h` — 1 gate

`#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)` at line 6, `#endif` at line 141. Wraps the entire file. Delete the wrapper.

#### `runtime/include/nemotron/p15_smem_partition_generated.h` — 1 gate

```cpp
#if defined(__CUDACC__)
#define NEMOTRON_P15_SMEM_HD __host__ __device__
#else
#define NEMOTRON_P15_SMEM_HD
#endif
```
nvcc is the only compiler used; `__CUDACC__` is always defined. Replace with `#define NEMOTRON_P15_SMEM_HD __host__ __device__`. **If this file is emitted by a generator script, fix the generator too — otherwise the gate reappears on next regeneration.** Investigate before editing.

#### `runtime/include/nemotron/p15_probe_generated.h` — 1 gate

Same pattern as above for `NEMOTRON_P15_PROBE_HD`. Same generator caveat.

#### `runtime/src/loader/artifact_loader.cpp` — 1 gate

`#ifdef POSIX_FADV_SEQUENTIAL` at line 164 → 166. Always defined on Linux 2.5.60+; target is Linux. Delete the gate, keep the body.

#### `testing/api/nano_save_prompt_oracle.cpp` — 4 gates

Platform gates (`_WIN32` / `__linux__`) at lines 27, 128, 154, 429. Target is Linux-only; delete `_WIN32` branches, keep `__linux__` branches.

#### CMake macro sources

- `runtime/CMakeLists.txt:197-201`: unconditionally sets `NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE=1` on the `nemotron_runtime_backend` library target (the target that compiles `fused_moe_prefill.cu` per line 149 of the same file). Delete the `target_compile_definitions` block.
- `testing/CMakeLists.txt:314-318, 349-353, 384-388`: sets the same macro on three test targets (`p15_swizzled_pipeline_test`, `p5_swizzled_pipeline_test`, `p5_swizzled_pipeline_test_120f`) guarded on `NEMOTRON_TEST_LOCAL_CUTLASS_INCLUDE_DIR`. Delete the three `target_compile_definitions` blocks.
- `NEMOTRON_TEST_LOCAL_CUTLASS_INCLUDE_DIR`: also guards `target_include_directories` blocks on those test targets. Grep repo-root `CMakeLists.txt` for its definition. If it also guards a legitimate include-path override, keep the include-dir portion and delete only the compile-definitions portion. If it exists solely to gate the now-deleted macro, delete it entirely.

#### Clean files (no gates outside include guards)

- `runtime/include/nemotron/*.h` — all files besides the four listed above.
- All `testing/backend/*.{cpp,cu}` (except `nano_save_prompt_oracle.cpp` which is in `testing/api/`).
- All `benchmarks/**/*.{cpp,cu}`.

### `fused_moe_prefill.cu` structural facts

- **Namespace layout:** `namespace nemotron {` at line 48, `namespace {` (anonymous) at line 58, `namespace nvfp4_cute {` 110→122, `namespace nvfp4_bridge {` 125→1939, anonymous continues through 12849, `}  // namespace` 12849, `}  // namespace nemotron` 13716.
- **No file-scope `static` free functions.** Only `static constexpr int kScaleVecSize = 16;` at line 114 and 64 other `static constexpr` class-member constants. No scope-crossing `static` state.
- **No `extern "C"` or `extern template`.**
- **Forward declarations centralized at lines 71-105** for `ExecutionScaleOffset`, `RoundUp`, `LaunchZeroBuffer`, `LaunchZeroBf16Buffer`, `LaunchGatherRows`, `LaunchRoutedBf16Relu2Pack`, `LaunchPlannedPackedInputMatVecBf16`. Fragments can rely on these being declared before any definition.
- **Includes at lines 1-46** — all top-of-file, no mid-file `#include`s.
- **No `__LINE__` / `__FILE__` usage** in the file.

### Public interface

- `runtime/include/nemotron/fused_moe_prefill.h` declares the public symbols including `RunNanoP1KernelForTesting` (line 114) and `RunNanoP1DirectPackKernelForTesting` (line 127). Test binaries link against these symbols via `target_link_libraries(... PRIVATE nemotron_runtime_backend)`; they do not `#include` the `.cu` file. **The umbrella split is internal — no test file needs modification.**

### Line-number references in plan documents

- Do **not** trust a hard-coded count here; re-run the grep at execution time and treat the result as authoritative.
- Current review found live references in `proj-2026-04-12-1022/PLAN.md`, active NanoP1 probe notes under `proj-2026-04-12-1022/probes/`, `proj-2026-04-11-1554/PLAN.md`, and this cleanup plan itself.
- Historical documents to leave alone remain: `PLAN_v4_abandoned.md`, `proj-2026-04-10-*`, `proj-2026-04-11-0400`, `proj-2026-04-05-*`, `proj-2026-04-04-0133`, `proj-2026-04-03-*`.

### Session WIP

- Working tree at plan authoring: `M runtime/src/backend/fused_moe_prefill.cu`, `M testing/backend/nano_p1_direct_pack_oracle_test.cpp`, `M testing/backend/nano_p1_mainloop_oracle_test.cpp`, plus several benchmark and tool files.
- The stopped session is `86e37198`; its WIP has not been committed.
- The most recent compaction summary (from the killed session) contains empirically-verified derived facts (thread→(M,N) mapping, `N_mf_table`, CLayout column-major formula) worth preserving before the WIP is discarded.

## Pre-work

### P.1 — Capture session-derived facts into a cheat sheet

**Goal:** Preserve the empirically-verified layout facts from the killed session's final compaction summary before discarding the WIP.

**Action:** Read `~/.claude/projects/-home-khkramer-src-nemotron-inference/86e37198-c2d2-460f-8096-61c13808afb6.jsonl`, specifically the compaction summary at timestamp `2026-04-14T00:47:44Z` (the most recent). Extract only DIAG-confirmed and compile-probe-confirmed facts:

- `TracedP5PermTileN` = `Layout<Shape<_8, _2, _2>, Stride<_1, _16, _8>>` and its flat-index mapping: `{0..7}→{0..7}`, `{8..15}→{16..23}`, `{16..23}→{8..15}`, `{24..31}→{24..31}`.
- `SM80_16x8_Row` CLayout = `Layout<Shape<Shape<_4,_8>,Shape<_2,_2>>, Stride<Stride<_32,_1>,Stride<_16,_8>>>`. Empirically column-major (M-fast): `M = flat % 16`, `N = flat / 16`.
- Thread-to-(M,N) mapping for `NanoP1TiledMma`: `atom_m = (t % 128) / 32`, `atom_n = t / 128`, `t_local = t % 32`, `M = atom_m*16 + (t_local/4) + (reg/2)*8 + nf*64`, `N = (t_local % 4)*2 + (reg % 2) + N_mf_table[mf]`.
- `N_mf_table` for `atom_n=0`: `{0, 8, 32, 40, 64, 72, 96, 104}`.
- NanoP1 `AccumLayout` shape = `(4, 2, 8)` where dimensions represent `(reg, nf, mf)`; `mf` (labeled "M fragment") actually controls N and `nf` (labeled "N fragment") actually controls M, due to `TracedP5PermTileN` in `ValLayoutMNK`.
- Phase 3 mismatch pattern: matches occur only at `M%8 ∈ {0,2} ∧ N%8 ∈ {0,2}` (~6.25% match rate, 230,087 / 245,760 mismatches).
- **Ruled out** (do not re-derive):
  - `sm120_rr_smem_copy_selector_A/B` with `UseF8f6f4=true` — identical SrcLayout/DstLayout to `UseF8f6f4=false`.
  - `SM80_16x8_Row` as row-major — DIAG empirically confirms column-major.

**Do not include** the flip-flopping hypotheses (partition_fragment convention swap, do_shuffle row permutation, etc.) — those were session guesses, not verified facts, and some of them made the mismatch count worse before being reverted.

**Output:** `proj-2026-04-12-1022/NANO_P1_LAYOUT_CHEATSHEET.md` (under the active project dir, not this plan's dir, so it's co-located with the NanoP1 work).

**Verification:** File exists, content is ≤150 lines, all claims are traceable to DIAG output or compile-time probes in the killed session's transcript.

**Commit:** "docs: capture NanoP1 empirical layout facts from session 86e37198" (adds the cheat sheet on a clean working tree — but see P.2 for sequencing).

### P.2 — Checkpoint session WIP to a side branch

**Goal:** Preserve the killed session's uncommitted edits as a recoverable git reference **without** sweeping unrelated dirty-tree work into the checkpoint, then continue the cleanup from a clean execution worktree.

**Action:**
```
git status --short

# Create a clean execution worktree from HEAD. All cleanup commits happen there.
git worktree add ../nemotron-inference-proj-2026-04-13-1936 -b proj-2026-04-13-1936-exec HEAD

# In that clean worktree, create the checkpoint branch for the killed session.
git -C ../nemotron-inference-proj-2026-04-13-1936 switch -c session-86e37198-wip

# Copy only the files that indisputably belong to session 86e37198.
# Minimum expected set:
cp runtime/src/backend/fused_moe_prefill.cu \
   ../nemotron-inference-proj-2026-04-13-1936/runtime/src/backend/fused_moe_prefill.cu
cp testing/backend/nano_p1_direct_pack_oracle_test.cpp \
   ../nemotron-inference-proj-2026-04-13-1936/testing/backend/nano_p1_direct_pack_oracle_test.cpp
cp testing/backend/nano_p1_mainloop_oracle_test.cpp \
   ../nemotron-inference-proj-2026-04-13-1936/testing/backend/nano_p1_mainloop_oracle_test.cpp
# If P.1 already created the cheat sheet, or the session produced probe/notes files
# that must be preserved, copy those explicit paths one by one as well.

git -C ../nemotron-inference-proj-2026-04-13-1936 add -- \
  runtime/src/backend/fused_moe_prefill.cu \
  testing/backend/nano_p1_direct_pack_oracle_test.cpp \
  testing/backend/nano_p1_mainloop_oracle_test.cpp
# If additional copied files were preserved, add them explicitly in the same
# checkpoint commit rather than broadening the add command.
git -C ../nemotron-inference-proj-2026-04-13-1936 commit -m \
  "Checkpoint: session 86e37198 WIP (no fix landed, reference only)"

# Return the clean execution worktree to its execution branch.
git -C ../nemotron-inference-proj-2026-04-13-1936 switch proj-2026-04-13-1936-exec

# Configure a fresh build tree in the execution worktree. Do not reuse the
# original worktree's build/ directory: CMake caches the source path in
# CMAKE_HOME_DIRECTORY, so the original build tree remains bound to the
# original checkout.
cmake -S ../nemotron-inference-proj-2026-04-13-1936 \
      -B ../nemotron-inference-proj-2026-04-13-1936/build \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Hard rules:
- Do **not** use `git add -A`.
- Do **not** commit unrelated benchmark/tool/doc edits just because they are present in the original worktree.
- Leave the original dirty worktree alone; the cleanup plan executes in `../nemotron-inference-proj-2026-04-13-1936`.

After this, `git -C ../nemotron-inference-proj-2026-04-13-1936 show session-86e37198-wip` recovers the killed session's changes, and the execution worktree on `proj-2026-04-13-1936-exec` is clean and matches HEAD.

**Verification:** `git -C ../nemotron-inference-proj-2026-04-13-1936 status` on `proj-2026-04-13-1936-exec` is clean, `git -C ../nemotron-inference-proj-2026-04-13-1936 log session-86e37198-wip -1` shows the checkpoint commit, and `../nemotron-inference-proj-2026-04-13-1936/build/compile_commands.json` exists with `CMAKE_HOME_DIRECTORY` pointing at `../nemotron-inference-proj-2026-04-13-1936`.

**No commit on the execution branch yet.** The checkpoint commit lives only on `session-86e37198-wip`.

**Execution root from here on:** all subsequent shell snippets assume cwd `../nemotron-inference-proj-2026-04-13-1936` unless they explicitly use `git -C ...` or another path prefix.

### P.3 — Add the cheat sheet on the clean branch

**Goal:** Get the cheat sheet into `proj-2026-04-13-1936-exec` without the WIP.

**Action:** Recreate the cheat sheet authored in P.1 (it was written while WIP was still in the tree; if it was copied during P.2, it now exists only on `session-86e37198-wip`, not on `proj-2026-04-13-1936-exec`). Either cherry-pick just the cheat-sheet file from the side branch, or re-author it from the session transcript. Commit.

**Commit:** "docs: capture NanoP1 empirical layout facts from session 86e37198"

**Verification:** `git show HEAD -- proj-2026-04-12-1022/NANO_P1_LAYOUT_CHEATSHEET.md` shows the content.

### P.4 — Re-baseline Phase 3 mismatch count

**Goal:** Establish the post-WIP-discard Phase 3 baseline so Step 0 can verify equivalence against it.

**Action:**
```
cmake --build build --parallel $(nproc)
./build/testing/nano_p1_mainloop_oracle_test 2>&1 | tee /tmp/phase3_baseline.log
```

Record the Phase 3 mismatch count. It may or may not equal 230,087 depending on what the session had done to the file before it was killed. Whatever it is, it's the new baseline.

**Verification:** baseline recorded; subsequent steps must match it.

## Step 0 — Delete all preprocessor gates

**Scope:** 8 source files + 2 CMake files. ~79 gate occurrences total.

**Sequencing within Step 0:** edit one file, compile, and spot-check with `nvcc -E` before moving to the next. Commit at the end of the step, not per-file — but the working-tree progression is incremental so debugging is local if something breaks.

### 0.1 — `runtime/src/backend/fused_moe_prefill.cu`

For each `NEMOTRON_P5_PARTITION_DEBUG` block (4 occurrences — lines 1892, 1979, 5599, 8491): delete the `#if` line, the body, and the `#endif` line. The macro is never defined anywhere in the repo (verified by grep); the body is unconditionally dead code.

For each `NEMOTRON_P5_LINEAR_PARTITION_DEBUG` block (4 occurrences — lines 1897, 1983, 5638, 8515): same. Delete entirely.

For each `NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE` block (~51 occurrences): keep the `#if` branch body, delete the `#else` branch body (if present), delete the `#if`, `#else`, and `#endif` lines. The macro is unconditionally set via `target_compile_definitions` wherever the file is compiled (see CMake section below), so the `#if` branch is always the live path.

For each `__CUDA_ARCH__` gate (3 pairs at 3771/3773, 3892/3894, 4201/4226): keep the `#if` branch (SM120 satisfies both `>= 900` and `>= 1000`), delete fallback branches, delete scaffolding.

**Caution:** some gate pairs are nested (e.g., line 7058 opens a `LOCAL_CUTE` block that contains `NEMOTRON_P5_PARTITION_DEBUG` blocks at 8491 and 8515 before its own `#else` at 8582). When deleting the inner gates first, the outer gate's line numbers shift. Work outside-in, or use a script that tracks offsets, or delete all gates in a single pass with a script that parses `#if`/`#endif` pairing.

**Recommended approach:** write a small Python script that reads the file, tracks `#if`/`#endif` depth, identifies each gate by its opening macro name, and emits the file with specified gates inlined. Save the script under `proj-2026-04-13-1936/scripts/` for reproducibility. Do not use `sed` — it can't handle nested conditionals safely.

Expected delta: ~500-700 lines removed in this file alone.

### 0.2 — `runtime/src/backend/device_nvfp4_matrix.cu`

Inline all 7 `NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE` gates at lines 18, 852, 926, 1012, 1086, 1172, 1245. Same policy: keep `#if` branch, delete `#else` if present, delete scaffolding.

### 0.3 — `runtime/include/nemotron/p15_scale_helpers.h`

Delete the file-wide `#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)` wrapper at line 8 and its matching `#endif` at line 679. Keep the body unchanged.

### 0.4 — `runtime/include/nemotron/p15_scale_runtime_helpers.h`

Same as 0.3 for the wrapper at lines 6 and 141.

### 0.5 — `runtime/include/nemotron/p15_smem_partition_generated.h`

**Before editing:** grep the repo for a generator script that emits this file. Search for `p15_smem_partition_generated`, `gen_p15`, or a CMake custom target that writes to this path. Also check the top of the file for a "DO NOT EDIT — GENERATED BY ..." comment.

- **If a generator exists:** fix the generator to emit the unconditional form, then re-run it (or update the file to match the new generator output).
- **If no generator exists** (the `_generated` suffix is vestigial): edit the file directly.

Replace the `__CUDACC__` gate (lines 7-11) with:
```cpp
#define NEMOTRON_P15_SMEM_HD __host__ __device__
```

### 0.6 — `runtime/include/nemotron/p15_probe_generated.h`

Same as 0.5, for `NEMOTRON_P15_PROBE_HD`.

### 0.7 — `runtime/src/loader/artifact_loader.cpp`

Delete the `#ifdef POSIX_FADV_SEQUENTIAL` gate at lines 164-166. Keep the call to `posix_fadvise(..., POSIX_FADV_SEQUENTIAL)` unconditional. The macro is defined on every Linux since kernel 2.5.60 (January 2003); the target platform is a modern Linux install on RTX 5090, so this is always defined.

Verify `<fcntl.h>` is already included unconditionally at the top of the file.

### 0.8 — `testing/api/nano_save_prompt_oracle.cpp`

Delete all 4 gates at lines 27, 128, 154, 429. Keep the Linux branches (`__linux__`), delete the Windows branches (`_WIN32`). Target platform is Linux-only.

### 0.9 — CMake

In `runtime/CMakeLists.txt`, delete lines 197-201:
```cmake
target_compile_definitions(
  nemotron_runtime_backend
  PRIVATE
    NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE=1
)
```

In `testing/CMakeLists.txt`, delete the three `target_compile_definitions` blocks that set the same macro at lines 314-318, 349-353, 384-388. Leave surrounding `if(NEMOTRON_TEST_LOCAL_CUTLASS_INCLUDE_DIR)` / `target_include_directories` intact pending audit.

**Audit `NEMOTRON_TEST_LOCAL_CUTLASS_INCLUDE_DIR`:** grep the repo for where it's defined and what else it gates. The three `if(...)` blocks in `testing/CMakeLists.txt` that wrapped the deleted `target_compile_definitions` also wrap `target_include_directories` blocks. If the variable exists solely to override the CUTLASS include path for tests (e.g., to point at a vendored copy vs. the default), keep the `target_include_directories` blocks. If it exists only to gate the now-deleted macro, delete the entire `if()` blocks. Decide based on what the grep shows.

### 0.10 — Verification

1. Refresh configure + clean build: `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build --parallel $(nproc)`.
2. Full test sweep: `ctest --test-dir build --output-on-failure -j1`. Expected: 68/71 pass (3 known pre-existing failures per `PLAN_RULES.md`). **Any new failure is a regression. Diagnose before commit.**
3. Focused MoE prefill regression: `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build/testing/nano_24_token_prefill_regression_test`. Expected: exact matches for prompt23 and prompt24, per `PLAN_RULES.md`.
4. Phase 3 baseline check: `./build/testing/nano_p1_mainloop_oracle_test 2>&1 | grep "Phase 3.*total_mismatches\|Phase 3.*PASS"` — mismatch count must equal the P.4 baseline exactly.
5. Preprocessor / compile-surface equivalence checks:
   - `runtime/src/backend/fused_moe_prefill.cu`: capture pre-edit `nvcc -E` output using the pre-edit `build/compile_commands.json`, then after Step 0 refresh configure and capture post-edit `nvcc -E` output using the post-edit `build/compile_commands.json`. Do **not** reuse the pre-edit command line after the CMake macro deletion; the point is to validate the real post-cleanup build surface. The diff must contain only the dead-branch lines that were intentionally deleted, modulo `#line` directives.
   - `runtime/src/backend/device_nvfp4_matrix.cu`: same.
   - Header-only gate removals (`p15_scale_helpers.h`, `p15_scale_runtime_helpers.h`, `p15_smem_partition_generated.h`, `p15_probe_generated.h`) are **not** required to have their own standalone `nvcc -E` command. If a live CUDA includer exists, use that includer as the representative preprocess surface; otherwise, use direct source diff + generator output diff (for the two `_generated.h` files) + the clean rebuild/test sweep as the authoritative proof.
   - Host-only files (`artifact_loader.cpp`, `nano_save_prompt_oracle.cpp`) are validated by their normal compiler invocations during the clean build; do not force them through the CUDA preprocess workflow.

Example equivalence check for `fused_moe_prefill.cu`:
```bash
# Before editing:
FLAGS_BEFORE=$(jq -r '.[] | select(.file | endswith("fused_moe_prefill.cu")) | .command' build/compile_commands.json | sed 's/ -c .*//')
eval "$FLAGS_BEFORE -E" > /tmp/fused_moe_prefill.before.i

# After editing, refresh compile_commands so CMake macro removals are reflected:
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
FLAGS_AFTER=$(jq -r '.[] | select(.file | endswith("fused_moe_prefill.cu")) | .command' build/compile_commands.json | sed 's/ -c .*//')
eval "$FLAGS_AFTER -E" > /tmp/fused_moe_prefill.after.i

# Compare (ignoring #line directives):
diff <(grep -v '^# ' /tmp/fused_moe_prefill.before.i) \
     <(grep -v '^# ' /tmp/fused_moe_prefill.after.i)
```

### 0.11 — Commit

```
cleanup: remove all preprocessor gates from nemotron-inference

Single-target codebase (Nemotron-3 Nano, RTX 5090/SM120, one CUDA version,
nvcc only) does not need config/debug/platform/arch gates. All #if blocks
were dead scaffolding for alternate configurations that don't exist.

Removed:
- 63 NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE gates (unconditionally defined in CMake)
- 8 NEMOTRON_P5_*_DEBUG blocks (never defined anywhere, dead code)
- 3 __CUDA_ARCH__ >= 900 / >= 1000 gates (SM120 always satisfies)
- 2 __CUDACC__ gates in generated headers (nvcc is the only compiler)
- 1 POSIX_FADV_SEQUENTIAL gate (always defined on Linux)
- 4 _WIN32 / __linux__ platform gates (Linux-only target)
- 4 target_compile_definitions macros from CMake

Zero logic changes. Binary behavior unchanged; nano_p1_mainloop_oracle_test
Phase 3 mismatch count identical to pre-commit baseline.
```

Expected delta: ~700-1000 lines removed across 8 source files + 2 CMake files. Zero lines added (except possibly the unconditional `#define` in the two `_generated.h` files).

## Step 1 — Split `fused_moe_prefill.cu` into umbrella + fragments

**Scope:** refactor `runtime/src/backend/fused_moe_prefill.cu` only. All other files untouched.

### 1.1 — Re-survey post-Step-0 file

After Step 0, re-run the namespace/function/`static` grep from the Investigation Facts section on the cleaned file. The line numbers will have shifted (~500-700 fewer lines) but the logical structure is unchanged. Use the post-Step-0 line numbers for the cut points.

### 1.2 — Design fragment layout

Target 10-11 fragment files in `runtime/src/backend/fused_moe_prefill/`. Proposed fragmentation (adjust after 1.1 re-survey):

| Fragment file | Contents | Approx. line count |
|---|---|---:|
| `common_helpers.cuh` | top-of-file forward declarations (current 71-105), shared constants, `ExecutionScaleOffset`, `RoundUp` helpers | ~200 |
| `nvfp4_cute.cuh` | `nvfp4_cute` named sub-namespace (current 110-122) | ~15 |
| `nvfp4_bridge_p5.cuh` | `nvfp4_bridge` entries for P5: types, helpers, scale partitions | ~400 |
| `nvfp4_bridge_p13_p15.cuh` | `nvfp4_bridge` entries for P13 and P15 | ~600 |
| `nvfp4_bridge_nano_p1.cuh` | `nvfp4_bridge` NanoP1 types (current 400-492) and helpers | ~300 |
| `p5_kernel.cuh` | `ComputeTracedP5*` mainloop, P5 epilogue variants | ~2,500 |
| `p13_p15_kernel.cuh` | P13 and P15 mainloops and epilogues | ~3,500 |
| `nano_p1_mainloop.cuh` | `ComputeNanoP1AccumTile` and its direct helpers (current ~12200-12510) | ~500 |
| `nano_p1_epilogue.cuh` | `StoreNanoP1CFragmentsRowMajor` and variants (current ~5146-5220) | ~200 |
| `internal_dispatch.cuh` | anonymous-namespace tail helpers near the file end: `RunNanoP1*Impl`, `P5NativeDirectPackOracleKernel`, and any remaining internal wrappers before the anonymous namespace closes | ~250 |
| `public_exports.cuh` | public `namespace nemotron` definitions that currently start after the anonymous namespace closes: `CopyP13DebugTrace`, profile selectors, `Run*ForTesting`, `RunFusedMoePrefill` | ~1,300 |

The sizes are rough; adjust after seeing the cleaned file. The critical invariants for the split:
- Each fragment is a contiguous line range in the current file (no reordering).
- Fragment boundaries align with logical function/struct definitions, not mid-function.
- Fragments containing `nvfp4_bridge::` symbols are wrapped by the umbrella with `namespace nvfp4_bridge { ... }`.
- Fragments included before the current anonymous-namespace close stay in the outer anonymous namespace.
- Public definitions that currently live after the anonymous-namespace close must stay outside it after the split. Do not pull exported functions into an internal-linkage fragment just to keep the umbrella shorter.

### 1.3 — Create the fragments directory and files

```
mkdir -p runtime/src/backend/fused_moe_prefill
```

For each fragment:
- Create the `.cuh` file with a top comment:
  ```cpp
  // Included exclusively from fused_moe_prefill.cu.
  // Do not #include this file elsewhere; no include guards.
  // Scope at point of inclusion: namespace nemotron::(anonymous)[::nvfp4_bridge]
  // Declared upstream in common_helpers.cuh: ExecutionScaleOffset, RoundUp, <etc.>
  ```
  For `public_exports.cuh`, change the scope comment to `namespace nemotron` because it is included after the anonymous namespace closes.
- Paste the contiguous line range from the current `fused_moe_prefill.cu`. Preserve whitespace and comments exactly.
- Do **not** add `#pragma once`. Do **not** wrap in `namespace {` — the umbrella owns scope.

### 1.4 — Rewrite `fused_moe_prefill.cu` as the umbrella

Target structure (~200 lines):

```cpp
// Umbrella TU for the MoE prefill kernels.
// Each .cuh below is a private fragment included exactly once from here.
// Do not add logic here — put it in a fragment.

#include "nemotron/fused_moe_prefill.h"
#include <cuda_bf16.h>
// ...all the <system> and <cutlass/...> includes from the current top of file...
#include "fused_decode_common.cuh"
#include "routed_p5_tma_descriptor.cuh"

namespace nemotron {

bool RunLaunchPlannedNvfp4ExpertMatVecBf16(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output);

namespace {

#include "fused_moe_prefill/common_helpers.cuh"
#include "fused_moe_prefill/nvfp4_cute.cuh"

namespace nvfp4_bridge {
#include "fused_moe_prefill/nvfp4_bridge_p5.cuh"
#include "fused_moe_prefill/nvfp4_bridge_p13_p15.cuh"
#include "fused_moe_prefill/nvfp4_bridge_nano_p1.cuh"
}  // namespace nvfp4_bridge

#include "fused_moe_prefill/p5_kernel.cuh"
#include "fused_moe_prefill/p13_p15_kernel.cuh"
#include "fused_moe_prefill/nano_p1_epilogue.cuh"
#include "fused_moe_prefill/nano_p1_mainloop.cuh"
#include "fused_moe_prefill/internal_dispatch.cuh"

}  // namespace

#include "fused_moe_prefill/public_exports.cuh"

}  // namespace nemotron
```

Preserve the original top-of-file includes exactly. Preserve the existing top-level forward declaration that currently lives between `namespace nemotron {` and `namespace {`; it cannot move into an anonymous-namespace fragment. Do not add new `using`s or aliases.

### 1.5 — Update `runtime/CMakeLists.txt`

Add all fragment files to `target_sources(nemotron_runtime_backend PRIVATE ...)`. Note: they compile as part of `fused_moe_prefill.cu`'s TU (they're `#include`d), so CMake doesn't compile them separately — but adding them to `target_sources` ensures:
- CMake tracks them as dependencies (editing a fragment triggers a rebuild of the TU).
- IDEs and `compile_commands.json` consumers see them as part of the build.

```cmake
target_sources(
  nemotron_runtime_backend
  PRIVATE
    src/backend/fused_moe_prefill.cu
    src/backend/fused_moe_prefill/common_helpers.cuh
    src/backend/fused_moe_prefill/nvfp4_cute.cuh
    src/backend/fused_moe_prefill/nvfp4_bridge_p5.cuh
    src/backend/fused_moe_prefill/nvfp4_bridge_p13_p15.cuh
    src/backend/fused_moe_prefill/nvfp4_bridge_nano_p1.cuh
    src/backend/fused_moe_prefill/p5_kernel.cuh
    src/backend/fused_moe_prefill/p13_p15_kernel.cuh
    src/backend/fused_moe_prefill/nano_p1_epilogue.cuh
    src/backend/fused_moe_prefill/nano_p1_mainloop.cuh
    src/backend/fused_moe_prefill/internal_dispatch.cuh
    src/backend/fused_moe_prefill/public_exports.cuh
)
```

(Adjust to match the existing CMake style for the target; if the main source list is elsewhere, merge the additions into that location.)

### 1.6 — Verification

1. `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build --parallel $(nproc)`. First build is a cache miss (different header set). Expect 1-2 minute build of the umbrella TU. Subsequent edits to a single fragment should rebuild only the umbrella TU in similar time.
2. `ctest --test-dir build --output-on-failure -j1`. Expected: 68/71 (same as Step 0).
3. `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build/testing/nano_24_token_prefill_regression_test`. Expected: exact matches for prompt23 and prompt24, same as Step 0.
4. Phase 3 mismatch count: unchanged from Step 0 baseline.
5. Preprocessor equivalence: `nvcc -E` of the post-Step-1 `fused_moe_prefill.cu` against the post-Step-0 version (with all fragment `#include`s expanded). Diff should be empty modulo `#line` directives.
6. Symbol equivalence: `nm -C build/.../fused_moe_prefill.cu.o` before and after Step 1. Symbol list should be identical.

### 1.7 — Commit

```
refactor: split fused_moe_prefill.cu into umbrella + fragments

Moves ~13,000 lines of kernel code from a single .cu file into multiple
.cuh fragments under fused_moe_prefill/. The .cu file is now a ~200-line
umbrella that #includes the fragments in original order while preserving
the original anonymous-namespace boundary and public-export scope.

Zero logic changes. Single TU, no -rdc=true, cross-function inlining
preserved. Symbol list unchanged. Test results unchanged.

Motivation: Claude Code's compaction auto-reload was pulling the full
13k-line .cu file on every compaction, consuming ~137k tokens of
context and forcing the 2026-04-13 NanoP1 debugging session to
re-compact every ~30 minutes. Fragments let the compaction reload
attach only the edited file, reducing per-compaction cost from ~147k
to ~20k tokens.
```

## Step 2 — Rewrite line-number references to symbol references

### 2.1 — Grep active plans

```
grep -rn 'fused_moe_prefill\.cu:[0-9]' \
  proj-2026-04-12-1022/ \
  proj-2026-04-11-2015/ \
  proj-2026-04-11-1554/ \
  proj-2026-04-13-1936/PLAN.md
```

Do **not** hard-code an expected hit count. The grep output at execution time is authoritative; use it as the starting set, then filter out the historical files listed in 2.3 before editing.

### 2.2 — Rewrite each reference

For each `fused_moe_prefill.cu:<N>` or `fused_moe_prefill.cu:<N>-<M>` reference:
1. Determine which symbol the line range corresponds to (use the Step 1 fragment layout table as a map).
2. Rewrite to `fused_moe_prefill/<fragment>.cuh::<symbol>`. Example: `fused_moe_prefill.cu:12200-12510` → `fused_moe_prefill/nano_p1_mainloop.cuh::ComputeNanoP1AccumTile`.
3. If the reference was to a comment or non-symbol line, rewrite to the nearest containing symbol.
4. Rewrite this cleanup plan's own example references too; Step 2 is incomplete if `proj-2026-04-13-1936/PLAN.md` still contains `fused_moe_prefill.cu:<line>` after the step.

### 2.3 — Leave abandoned plans alone

Files NOT to touch:
- `proj-2026-04-12-1022/PLAN_v4_abandoned.md` (33 refs)
- `proj-2026-04-10-*/**/*.md`
- `proj-2026-04-11-0400/**/*.md`
- `proj-2026-04-05-*/**/*.md`
- `proj-2026-04-04-0133/**/*.md`
- `proj-2026-04-03-*/**/*.md`

These are historical documents. Make-work to rewrite.

### 2.4 — Commit

```
docs: rewrite fused_moe_prefill.cu line refs to symbol refs in active plans

Symbol refs survive future line renumbering and future file splits.
Abandoned plans (PLAN_v4_abandoned.md, proj-2026-04-0[3-5]*, etc.)
left alone — historical documents don't need maintenance.
```

## Step 3 — Validate whether the Claude Code context-savings premise actually holds

**Not a code change step.** This step tests the assumption underlying the whole refactor. Treat the compaction heuristic as unknown until the log proves otherwise.

### 3.1 — Start a fresh Claude Code session

On `proj-2026-04-13-1936-exec` post-Step-2, start `claude --dangerously-skip-permissions` in `../nemotron-inference-proj-2026-04-13-1936`.

### 3.2 — Touch a fragment with a trivial edit

Make a whitespace-only edit to `runtime/src/backend/fused_moe_prefill/nano_p1_mainloop.cuh` (e.g., add and remove a blank line), and explicitly `Read` that fragment in the new Claude Code session. The goal is to give the session a fragment-sized context anchor without assuming why `compact_file_reference` chooses a file.

### 3.3 — Force a compaction

Let the session consume context via some exploration, or invoke `/compact` if available in the current Claude Code version.

### 3.4 — Inspect the compaction state

Read the new session's `.jsonl` file under `~/.claude/projects/-home-khkramer-src-nemotron-inference/<new-session-id>.jsonl` and filter for `compact_file_reference` attachments:

```bash
jq -c 'select(.attachment.type=="compact_file_reference") | .attachment.filename' \
   ~/.claude/projects/-home-khkramer-src-nemotron-inference/<new-session-id>.jsonl
```

Record the actual attachment set after compaction. Interpret it as follows:
- **Best case:** `nano_p1_mainloop.cuh` appears and `fused_moe_prefill.cu` does not. The refactor's main premise is confirmed.
- **Partial win:** the fragment appears, but the umbrella TU also appears. Some context-locality benefit exists, but the full 7x savings claim is not yet proven.
- **Failure:** the umbrella TU still appears alone. The split may still help human navigation and targeted `Read` calls, but the auto-reload savings premise is not validated.

**If failure:** Claude Code's heuristic for `compact_file_reference` is something other than "recently-edited files" or explicit fragment reads. Investigate by examining other fields in the attachment record and by reading the Claude Code source if accessible. Mitigations if the premise is wrong:
- Manually `Read` the fragment at the start of each session as an explicit context anchor.
- Add a `CLAUDE.md` section documenting which fragment to open for NanoP1 work.
- Consider whether the original 13k-line file still gets attached; if so, the savings come only from Read calls in-session, not from post-compaction reloads.

### 3.5 — No commit

This is verification only.

## Step 4 — Resume NanoP1 Phase 3 debugging (out of scope for this plan)

**Not part of this plan.** Listed here only as the motivation for why the cleanup steps matter.

Once Steps 0-3 are complete, a new session can:
1. Open `proj-2026-04-12-1022/NANO_P1_LAYOUT_CHEATSHEET.md` and `runtime/src/backend/fused_moe_prefill/nano_p1_mainloop.cuh` as the initial context anchors.
2. Apply the `probe-then-decide` protocol per `CLAUDE.md`. The specific probe the killed session identified as next-needed: compile-time `ShowInt<>` extraction of `tCsA` and `tCsB` layouts after `partition_S` in `ComputeNanoP1AccumTile`, to determine the exact smem-position-to-accumulator mapping.
3. Use the cheat sheet's empirical thread→(M,N) mapping as the ground-truth reference for what the kernel *should* be reading from smem at each register slot.
4. Decide on a fix based on the probe output — not on another round of speculative hypotheses.

This step is left to the session that does it; it is not prescribed here.

## Risks and mitigations

1. **Generator script for `_generated.h` files re-emits the old gate form.** Mitigation: find and fix the generator in Step 0.5/0.6, or confirm no generator exists and edit the files directly. Either way, the first post-cleanup `cmake --build` should surface any issue.

2. **Binary behavior changes despite mechanical edits.** Possible if a `#else` branch was secretly being compiled due to a forgotten `-D` flag somewhere, or if a deleted debug block had side effects beyond printing. Mitigation: the `nvcc -E` equivalence check in Step 0.10 catches this — if the preprocessed output diff has anything beyond the intended deletions, stop and investigate.

3. **Step 1 fragment split introduces ODR violations.** Unlikely because each fragment is included exactly once, the scope is the same anonymous namespace, and there are no `static` free functions to collide. But if Step 1 produces linker errors, the symbol-equivalence check in 1.6 should flag it pre-commit.

4. **Step 3 premise fails.** Steps 0-2 still have independent value (smaller file, easier grep, no rotting dead code, plans stay current) but the expected 7× context-savings doesn't materialize. Mitigation: acceptable outcome; steps are not wasted.

5. **Fragment granularity is wrong.** Fragments may still be too big (e.g., `p13_p15_kernel.cuh` at ~3,500 lines) or the split may bisect a function. Mitigation: the design in Step 1.2 is a starting point; adjust after seeing the post-Step-0 file structure. If a second split pass is needed later, it's a fast follow-up, not a rework.

6. **Step 0 cleanup touches the grandfathered transitional surfaces** (`TracedP5*`, `TracedP13*`, `TracedP15*`). Mitigation: the `PLAN_RULES.md` transitional-surface clause explicitly allows "mechanical renames, comment cleanup, and bug fixes that do not change the kernel structure". Gate removal is mechanical cleanup, not structural change. Document this reasoning in the Step 0 commit message.

## What Codex should look for in review

This is a factual-review request, not a should-we-do-this request. The user has already decided to proceed. Codex should check:

1. **Completeness of the gate inventory.** Is there any `#if` / `#ifdef` / `#ifndef` in the repo that this plan misses? Re-run the grep patterns in the Investigation Facts section and compare the output against the tables in Step 0. Report any file not listed.

2. **Correctness of file paths and line numbers.** The line numbers in this plan were captured during plan authoring and may shift if the working tree changes before execution. Verify against the current HEAD.

3. **Correctness of the CMake edits.** Is the `target_compile_definitions` block for `NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE` at exactly the lines claimed in `runtime/CMakeLists.txt`? Does `testing/CMakeLists.txt` have other uses of the macro beyond the three test targets listed? Is `NEMOTRON_TEST_LOCAL_CUTLASS_INCLUDE_DIR` used anywhere else?

4. **Generator-script caveat for `_generated.h` files.** Is there actually a generator? Where? What does it output? Is the gate pattern in the generator template or hand-added after generation?

5. **Namespace structure claims for `fused_moe_prefill.cu`.** Verify: one outer `nemotron::`, one anonymous namespace, two named sub-namespaces (`nvfp4_cute`, `nvfp4_bridge`). No other nested namespaces. No scattered `namespace { }` blocks.

6. **Fragment boundary proposals in Step 1.2.** Are the proposed cut points reasonable? Are any of them likely to bisect a function, struct, or logical grouping that should stay together? Is there a better cut point within ~100 lines of the proposed one?

7. **Public ABI invariance.** Confirm that `runtime/include/nemotron/fused_moe_prefill.h` does not change, and that no test file or other `.cpp` file `#include`s `fused_moe_prefill.cu` directly (only via the header and linking).

8. **Test protocol compliance.** Does this plan's verification criteria match `PLAN_RULES.md`? Specifically: full ctest sweep at every step, Phase 3 baseline check, the 68/71 expected pass count.

9. **Hidden gates in generated files.** Beyond `p15_smem_partition_generated.h` and `p15_probe_generated.h`, are there other files with `_generated` in their name that might contain gates this plan doesn't cover?

10. **Commit granularity and messages.** Is the commit message style consistent with the repo's existing history (`git log --oneline | head -30`)? Are the commits atomic and independently revertible?

## References

- `CLAUDE.md` — Development workflow, probe-then-decide protocol.
- `PLAN_RULES.md` — Kernel provenance, safety, test protocol, pre-existing failures.
- `proj-2026-04-12-1022/PLAN.md` — The active NanoP1 plan whose Step 5 work was blocked by the context-thrash session.
- `proj-2026-04-12-1022/NANO_P1_LAYOUT_CHEATSHEET.md` — To be created in P.1; the empirically-verified layout facts from session `86e37198`.
- `~/.claude/projects/-home-khkramer-src-nemotron-inference/86e37198-c2d2-460f-8096-61c13808afb6.jsonl` — The session transcript; source for P.1's cheat sheet extraction.
- Killed session metadata: started 2026-04-13 16:17 UTC, ended 2026-04-14 00:54 UTC, 16 compactions, 0 commits, Phase 3 mismatch trajectory 230,087 → 243,611 → 230,087.
