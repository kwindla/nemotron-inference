# Step 2b — TRT-LLM BF16 `gemm1_output` capture harness

Plan v6 step 2 calls for "the smallest possible TRT-LLM runtime artifact that
gives step 4b a true BF16 boundary oracle." This directory contains the chosen
runtime path, the harness, the patch, and will eventually contain the captured
artifact itself under `golden/`.

## Chosen runtime path

We do NOT link against the vendored `libtensorrt_llm.so` (which would require
pulling torch / cublas into the Nemotron CMake link graph — see NOTES.md §5).

Instead, we drive **flashinfer's vendored copy of TRT-LLM's MoE GEMM kernel
set**, which is JIT-compiled from sources under
`<venv>/lib/python3.12/site-packages/flashinfer/data/csrc/` and exposed to
Python through the low-level `fused_moe_runner.run_moe(...)` binding (NOT the
high-level `cutlass_fused_moe(...)` wrapper — see "Tactic selection" below).
On SM120 this compiles the same `moe_gemm_tma_ws_launcher.inl` kernel
instantiations we want to match bitwise in step 4b.

### Venv auto-detect

The warm flashinfer JIT cache at
`~/.cache/flashinfer/0.6.6/120a/cached_ops/fused_moe_120/build.ninja` bakes
absolute source paths into every `cuda_compile` line. `run_capture.sh`
parses one of those lines, extracts the venv-rooted
`<venv>/lib/python3.12/site-packages/flashinfer/data/csrc` prefix, and uses
THAT venv's copy of `cutlass_fused_moe_kernels.cuh` as the patch target.
It then runs `${venv}/bin/python capture_bf16_gemm1.py ...`, so the Python
import resolves to the flashinfer package the cache was built against, and
ninja's incremental rebuild sees our patched source and recompiles only the
affected .o files. `run_capture.sh` fails loudly if the warm cache references
more than one distinct source tree, or if no warm cache exists — in the
latter case, run `flashinfer.fused_moe.cutlass_fused_moe(...)` once from
whichever venv should own the capture so the cache exists, then re-run.

On the current environment, the warm cache is wired to
`vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc`, not the
`.venv-trtllm` tree. The auto-detect handles this transparently — there is
no hardcoded venv path in the scripts.

## What gets captured

We add a minimal env-gated hook to flashinfer's copy of TRT-LLM's
`CutlassMoeFCRunner<T, WeightType, OutputType, ...>::gemm1()` (the TMA warp-
specialized path, NVFP4 → BF16 code path) that copies `gemm_output` to host
and writes it to a file plus a 64-byte header. The hook fires AFTER the
`moeGemm` call finishes and BEFORE `doActivation` transforms the BF16 into
packed FP4. `gemm_output` is the true BF16 pre-activation boundary: row-major
dense `[expanded_num_rows, fc1_out_size]`.

Exact patch: `patches/flashinfer_bf16_gemm1_dump.patch` — a unified diff
against the flashinfer `data/csrc` subtree, two small hunks (one `#include`
addition, one 42-line dump block). It applies cleanly against whichever
flashinfer install the warm JIT cache is wired to; `run_capture.sh` picks
the target tree at runtime (see "Venv auto-detect" above).

Dump file layout (64-byte header + body):

    off  len  field
    0    8    magic "NEMOP1\0\0"
    8    4    version (uint32, = 1)
    12   4    expanded_num_rows (uint32)
    16   4    fc1_out_size (uint32, elements, i.e. N)
    20   4    element_bytes (uint32, = 2 for bfloat16)
    24   4    is_gated_activation (uint32, 0 or 1)
    28   36   zero-pad (reserved)
    64   ..   raw bytes of gemm_output, row-major,
              stride = fc1_out_size elements per row

## Safety of the patch

- Exactly one file modified: `cutlass_fused_moe_kernels.cuh` in the installed
  flashinfer wheel of whichever venv owns the warm cache (2 hunks, ~47 lines).
- The dump is env-gated on `NEMOTRON_TRTLLM_DUMP_GEMM1` — if that env var is
  not set, the hook is a single `getenv(...) → nullptr` check that skips at
  runtime. It does NOT affect behavior of any existing flashinfer-consuming
  code.
- `run_capture.sh` ALWAYS reverts the patch on exit (success or failure) via
  an EXIT trap, so the installed flashinfer tree is left in its original
  state after each run.
- The flashinfer JIT ninja build tracks source mtimes; `touch`ing the source
  file after the patch invalidates only the few `.o` files that include this
  header, keeping incremental rebuilds fast.
- `run_capture.sh` ALSO deletes stale artifacts from `golden/` before the run
  (`bf16_gemm1_tactic*.bin`, `bf16_gemm1_metadata.json`, `inputs.pt`,
  `final_moe_output.pt`), and the Python harness verifies every dump file's
  `mtime >= t0_of_this_run` so an old file cannot false-pass the existence
  check if the kernel ever silently fails to write a fresh one.

## Running the capture

```bash
cd /home/khkramer/src/nemotron-inference
bash proj-2026-04-12-1022/trtllm_reference/run_capture.sh
```

Output:

    proj-2026-04-12-1022/trtllm_reference/golden/
      bf16_gemm1_tactic0.bin        # CtaShape128x128x128B, SwapAB=false
      bf16_gemm1_tactic1.bin        # CtaShape128x128x64B,  SwapAB=false  (plan-v6 P1) *
      bf16_gemm1_tactic2.bin        # CtaShape128x256x64B,  SwapAB=false
      bf16_gemm1_tactic3.bin        # CtaShape256x128x64B,  SwapAB=false
      bf16_gemm1_tactic4.bin        # CtaShape128x128x128B, SwapAB=true
      bf16_gemm1_tactic5.bin        # CtaShape128x128x64B,  SwapAB=true
      bf16_gemm1_tactic6.bin        # CtaShape128x256x64B,  SwapAB=true
      bf16_gemm1_tactic7.bin        # CtaShape256x128x64B,  SwapAB=true
      bf16_gemm1_metadata.json      # tactic list, labels, swap_ab, shapes, mtimes, paths
      inputs.pt                     # harness inputs (hs, FP4 weights, scales)
      final_moe_output.pt           # diagnostic: post-FC2 output of last tactic

Each dump file has a 64-byte header (magic + version + shape) followed by
the raw BF16 body, row-major, stride = `fc1_out_size` elements per row. See
the "Dump file layout" section above.

\* Step 4b's oracle compares bitwise against `bf16_gemm1_tactic1.bin` — see
"Tactic selection" below.

**Observed convergence on the default small problem**: all 8 tactics produce
bitwise-identical BF16 dumps (confirmed with an instrumented build that also
logged the actually-selected `config.tile_config_sm120 / swap_ab` per call).
This is a math property of FP4×FP4 reductions rounded to BF16 at the
epilogue boundary — see `NOTES.md §5 "Observed convergence across tactics"`
for the full analysis. It is expected to break at larger K (e.g. ≥4096),
which step 5 of the plan leans on as a sharper kernel-identity test.

Problem-shape knobs (env vars):

    NEMOTRON_HARNESS_M     num_tokens       (default 128)
    NEMOTRON_HARNESS_K     hidden_size      (default 256)
    NEMOTRON_HARNESS_N     inter_size       (default 256)
    NEMOTRON_HARNESS_E     num_experts      (default 1)
    NEMOTRON_HARNESS_TOPK  top_k            (default 1)
    NEMOTRON_HARNESS_SEED  seed             (default 0xC0FFEE)

The defaults intentionally pick a tiny single-expert problem so that the
permutation stage is trivial (expert_first_token_offset = [0, num_tokens]),
making step 4b's reproduction straightforward. Non-default values are
supported for step 4b stress tests.

## Tactic selection: enumerate, do not rely on fallback

The harness bypasses `flashinfer.fused_moe.core.cutlass_fused_moe(...)` (and
its AutoTuner) entirely. It calls the low-level JIT-built module directly:

    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module
    raw_jit_module = gen_cutlass_fused_moe_sm120_module(use_fast_build=False).build_and_load()
    fused_moe_runner = raw_jit_module.init(bf16, torch.int64, bf16, False, False, False, False)
    for tactic_id in range(fused_moe_runner.get_gemm1_tactic_count()):
        os.environ["NEMOTRON_TRTLLM_DUMP_GEMM1"] = f".../bf16_gemm1_tactic{tactic_id}.bin"
        fused_moe_runner.run_moe(..., [tactic_id, 0], ...)

Note: `flashinfer.fused_moe.core.get_cutlass_fused_moe_module(...)` returns a
`SimpleNamespace(cutlass_fused_moe=...)` that only exposes the high-level
wrapper (`core.py:715`). The raw tvm-ffi module with `.init(...)` lives on
`gen_cutlass_fused_moe_sm120_module(...).build_and_load()`, which is what
the harness calls.

The explicit `[tactic_id, 0]` profile IDs map directly to
`mAllProfiles[tactic_id]` in the binding (see
`flashinfer_cutlass_fused_moe_binding.cu:807-820`). For SM120 FP4 grouped
GEMM, `mAllProfiles` is the SwapAB-duplicated ordered vector below:

| tactic_id | tile shape                          | swap_ab | plan v6 label |
|---|---|---|---|
| 0 | `CtaShape128x128x128B_Cluster1x1x1` | false   | — |
| 1 | `CtaShape128x128x64B_Cluster1x1x1`  | false   | **P1** (CTA_M=N=128, CTA_K=64 bytes, SwapAB=false) |
| 2 | `CtaShape128x256x64B_Cluster1x1x1`  | false   | — |
| 3 | `CtaShape256x128x64B_Cluster1x1x1`  | false   | — |
| 4 | `CtaShape128x128x128B_Cluster1x1x1` | true    | — |
| 5 | `CtaShape128x128x64B_Cluster1x1x1`  | true    | — |
| 6 | `CtaShape128x256x64B_Cluster1x1x1`  | true    | — |
| 7 | `CtaShape256x128x64B_Cluster1x1x1`  | true    | — |

Base tile shapes come from `cutlass_heuristic.cpp:601-616`. The SwapAB=true
variants are appended by `getTmaWarpSpecializedConfigs` in
`moe_gemm_template_dispatch.h:657-663`, which is why
`get_gemm1_tactic_count()` returns 8, not 4, and `get_gemm2_tactic_count()`
returns 16 (the same 8 plus their FINALIZE-fusion duplicates).

This is why the harness dumps every GEMM1 tactic rather than trusting
`tactic_id = -1`: the fallback resolves to `mAllProfiles.front()` =
`CtaShape128x128x128B`, which is NOT the tile shape plan v6 calls P1.
Step 4b's oracle reads the tactic whose `is_plan_v6_p1` flag in
`bf16_gemm1_metadata.json` is `true` (currently `tactic_id == 1`, labelled
`CtaShape128x128x64B_Cluster1x1x1`, SwapAB=false). The harness WARNs if
`get_gemm1_tactic_count()` changes from the expected 8, so step 4b can
re-verify the mapping against `cutlass_heuristic.cpp` +
`moe_gemm_template_dispatch.h` before trusting the label.

## Verifying a capture

After a run, quick sanity checks from Python:

```python
import numpy as np, pathlib, struct
p = pathlib.Path("proj-2026-04-12-1022/trtllm_reference/golden/bf16_gemm1_tactic1.bin")
hdr = p.read_bytes()[:64]
magic = hdr[0:8]
version, rows, cols, ebytes, gated = struct.unpack_from("<IIIII", hdr, 8)
assert magic.startswith(b"NEMOP1"), magic
assert version == 1
assert ebytes == 2  # bf16
body = np.frombuffer(p.read_bytes()[64:], dtype=np.uint16)
assert body.size == rows * cols
```

And from `bf16_gemm1_metadata.json`:

```python
import json, pathlib
meta = json.loads(pathlib.Path("proj-2026-04-12-1022/trtllm_reference/golden/bf16_gemm1_metadata.json").read_text())
p1 = [t for t in meta["captured_tactics"] if t["is_plan_v6_p1"]]
assert len(p1) == 1 and p1[0]["expected_tile_label"] == "CtaShape128x128x64B_Cluster1x1x1"
print("plan v6 P1 dump:", p1[0]["dump_path"])
```

## Files in this directory

- `README.md` — this file
- `NOTES.md` — step-4a-facing distillation of TRT-LLM P1 CollectiveBuilder /
  TiledMma contracts; §5 documents the step 2b harness path in depth
- `capture_bf16_gemm1.py` — per-tactic harness: builds inputs, calls
  `fused_moe_runner.run_moe(..., [tactic_id, 0], ...)` directly for each
  GEMM1 tactic, verifies dump mtime, writes metadata, saves tensors to
  `golden/`
- `run_capture.sh` — auto-detect venv from warm ninja cache → clean stale
  artifacts → apply patch → run harness → revert patch (always, via trap)
- `patches/flashinfer_bf16_gemm1_dump.patch` — unified diff against
  `<venv>/lib/python3.12/site-packages/flashinfer/data/csrc/
  fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` — runs cleanly
  against both `.venv-trtllm` and `vllm-env-cu128` copies of the file, but
  only the venv owning the warm cache is actually patched in a given run
- `golden/` — run output: per-tactic BF16 dumps, metadata JSON, reproducible
  input tensors
