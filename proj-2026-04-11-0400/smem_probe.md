# P5 Shared-Memory Probe

Command:

```bash
bash proj-2026-04-11-0400/run_p5_smem_probe.sh
```

Environment:

- GPU: `NVIDIA GeForce RTX 5090`
- Compute capability: `12.0`
- `sharedMemPerMultiprocessor = 102400 B`
- `sharedMemPerBlockOptin = 101376 B`

Measured P5 storage model:

- Current P5 mainloop components:
  - A storage: `32768 B`
  - B storage: `32768 B`
  - A scale storage: `8192 B`
  - B scale storage: `4096 B`
  - SFB TMA stage storage: `4096 B`
  - barriers: `32 B`
- Aggregate current mainloop struct size: `82944 B`
- 128×128 FP32 staging tile: `65536 B`

Compiler/runtime-visible kernel usage:

- `current_separate`: `81956 B` static shared, `1` CTA/SM
- `aliased_union`: `83968 B` static shared, `1` CTA/SM
- `current_plus_stage`: compile failure

Naive extra-stage compile failure:

```text
ptxas error: ... uses too much shared data (0x24404 bytes, 0x18c00 max)
```

Interpretation:

1. The original plan claim that a 128×128 FP32 staging tile "fits within the existing mainloop footprint" as an extra shared allocation is false.
2. A union/alias-style storage model does fit under the SM120 per-block limit, but it is not literally zero-cost in this probe: it increases static shared from `81956 B` to `83968 B` while leaving occupancy at `1` CTA/SM.
3. Because current P5 is already a 1-CTA/SM kernel on this platform, an aliased full-tile staging design may be acceptable as a correctness/reference path, but it is not a clean "free reuse" story.
4. For an implementation intended to optimize TTFT/TPS incrementally, the preferred production direction remains warp/quad-local scale reduction and direct pack. Full-tile FP32 staging should be treated as a reference/fallback technique, not the assumed final design.
