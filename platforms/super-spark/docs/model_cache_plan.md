# Model Cache Plan

Serialize fully-prepared model weights to a single flat cache file so that subsequent startups can bypass all weight transformation, swizzling, per-tensor allocation, and first-use lazy materialization overhead.

IMPORTANT: v1 cache generation should be deterministic from the verified manifest and descriptor metadata. It should not require a synthetic all-expert warmup forward pass to populate expert weights.

IMPORTANT: the first implementation pass for this plan now exists, and it changed the next recommendation. The current execution-ready cache format is functionally working but not performance-viable on DGX Spark. The next cache pass should keep the deterministic writer/loader scaffolding while redesigning the payload toward packed checkpoint bytes plus cheap runtime conversion, not a 100+ GB execution-ready FP32 cache.

IMPORTANT: that cache redesign is coupled to the next throughput push. The cache plan now depends on four active runtime work items:
1. native packed scaled-FP8 / BF16 linear execution
2. device-side expert dispatch and merge with fewer sync points
3. fused or reduced-launch Mamba execution
4. fused or reduced-launch attention execution

These are not optional side optimizations. They are the path to a compact cache that does not regress TTFT on DGX Spark.

## Current Startup Breakdown

After lazy routed-expert materialization, the startup picture is:

| Phase | Time | Notes |
|---|---:|---|
| Environment bootstrap (manifest + mmap) | ~2.1s | Flat, not worth optimizing further |
| Model construction (non-expert weights) | ~14.0-15.1s | BF16/FP8 → FP32 device conversion, norm/bias uploads, attention/Mamba slice creation |
| First token decode (includes lazy expert materialization) | ~5.6-11.9s | Depends on which experts are materialized on first use |
| Warmed second token decode | ~0.73s | Best current steady-state reference on the same manifest path |
| **End-to-end first token** | **~27.9s** | Meets the <30s rollout target |

The first-token cost includes ~11.9s of lazy expert upload that only happens once — subsequent tokens reuse the materialized experts. But any new expert selection (different routing) triggers more lazy uploads during serving.

That makes model cache the right TTFT area to attack next, but the first measured cache design matters:

| Path | environment | model/build-or-load | decode | Total |
|---|---:|---:|---:|---:|
| Uncached cold first token | ~2.1s | ~14.0s | ~11.9s | ~27.9s |
| Uncached two-token run, token 0 | ~2.1s | ~14.7s | ~5.6s | ~22.4s |
| Uncached two-token run, token 1 | reused | reused | ~0.73s | ~0.73s |
| Cache-backed v1 execution-ready load | ~2.2s | ~113.7s | ~1.09s first, ~0.82s second | ~116.9s first |

So the current v1 cache payload is functionally correct but slower than the uncached path by a very large margin.

## What Goes Into The Cache

There are now two distinct cache designs to keep separate:

1. Desired target design:
   - deterministic, manifest-derived
   - stores original packed checkpoint bytes for dense/FP8 tensors
   - stores aligned execution-ready NVFP4 payloads where precomputation is actually expensive
   - aims for a ~67 GB cache

2. Current implemented v1 experiment:
   - deterministic, manifest-derived
   - stores execution-ready FP32 dense/scaled-FP8/embedding tensors
   - stores aligned NVFP4 payloads
   - measured payload: `102103740420` bytes across `41643` entries
   - works functionally after the Mamba cached-tensor-shape fix, but is too large and too slow to load

The desired design remains the correct target.

After a full warmup pass (construction + at least one forward call that touches all experts), the desired cache contents are:

**Non-expert weights (~6 GB on device, stored packed ~3 GB):**
- Embedding table (BF16 → FP32): 131072 × 4096, store as BF16
- LM head (BF16 → FP32): same, store as BF16
- Attention Q/K/V/O (8 layers, BF16 → FP32): store as BF16
- Mamba in_proj/out_proj (40 layers, FP8 → FP32): store as FP8
- MoE control tensors (gate, fc1_latent, fc2_latent, shared_up): store in original format
- Norm weights, biases: store as FP32 (small)

**NVFP4 expert weights (~70.5 GB, stored as-is with pre-swizzled scales):**
- 40 MoE layers × 512 experts × 2 projections
- Packed NVFP4 data: stored as-is
- Swizzled matmul block scales: stored as-is (the expensive CPU swizzle is pre-computed)
- Tensor scales: stored as-is
- Raw block scales: **omitted** — never read during execution, saves ~6.7 GB

**cublasLt heuristic cache (optional in v1, tiny):**
- Today the runtime-side `GemmHeuristicCache` is cheap and deterministic rather than an expensive discovered artifact.
- So cache serialization of heuristic entries is optional for v1 and should not block the weight-cache path.
- If real backend-discovered heuristic state is added later, extend the cache format then.

**Desired total cache size: ~67 GB** (vs 75 GB original checkpoint, vs 91 GB device footprint)

**Current measured execution-ready v1 size: ~102.1 GB**

## Cache Format

One file:

1. **Header** (JSON, small)
   - format version
   - source checkpoint revision hash
   - manifest content hash
   - model config hash
   - tensor count, total payload bytes
   - tensor index: name, offset, nbytes, dtype, shape, 256-byte aligned offset
   - heuristic cache entries

2. **Payload** (flat bytes, loadable directly to device)
   - all buffers 256-byte aligned within the file (cublasLt NVFP4 requirement)
   - desired design: packed-format dense weights (BF16/FP8), converted on device after load
   - current experimental implementation: expanded FP32 execution surfaces for dense/scaled-FP8/embedding tensors
   - pre-swizzled NVFP4 expert data, directly usable by cublasLt
   - no raw block scales stored

## Load Path

1. Open cache file, read header, validate checkpoint hash
2. One `cudaMalloc` for the full payload
3. Stream payload into device memory — on DGX Spark UMA, read directly into the `cudaMalloc`'d region if the driver supports it, otherwise chunked read + `cudaMemcpy`
4. Desired design only: device-side BF16/FP8 → FP32 conversion for non-expert weights (~0.09s at 273 GB/s)
5. Walk tensor index, create non-owning views:
   - `DeviceNvfp4Weight::CreateView` for expert weights (already execution-ready)
   - `DeviceDenseWeightFp32::CreateView` for dense weights (after conversion)
   - `UploadedLinearOp::CreateNvfp4View` / `CreateDenseView` for linear ops
6. Populate `GemmHeuristicCache` from the stored entries if present
7. Assemble `SingleTokenForwardModel` from views — no lazy materialization needed, all experts pre-resident

If cache is missing or hash mismatches: fall back to current construction + warmup, then write cache.

Current measured result for the implemented v1 execution-ready cache:
- `cache_load_ms ≈ 110417.674`
- `model_build_ms ≈ 113652.054` including view assembly
- `hot_0_ms ≈ 1087.325`
- `hot_1_ms ≈ 817.771`
- top-1 token still matches (`5130`)

Interpretation:
- the cache-backed forward path is functionally viable
- the current payload shape is not TTFT-viable
- the next cache pass should reduce payload size, not just optimize the existing 102 GB load path

## Save Path

Build the cache directly from the verified manifest and descriptor metadata:

1. For each dense / scaled-FP8 / embedding tensor:
   - target design: write the original packed checkpoint bytes into the cache payload
   - do not store the expanded FP32 execution copy
2. For each NVFP4 expert or shared-expert tensor:
   - write packed data as-is
   - precompute and write the swizzled matmul block scales
   - write tensor scales
   - omit raw block scales from the cache payload
3. Optionally serialize `GemmHeuristicCache` entries if that state becomes meaningful
4. Write header with tensor index and hashes
5. Optional: after a cache miss, generate the cache in the background while serving from the current manifest path
6. Optional validation pass:
   - compare cache-loaded weights or first-token behavior against the uncached path
   - but validation is not the same thing as populating the cache contents

The key change from the earlier sketch is that full expert coverage should come from deterministic cache generation over all manifest-backed expert descriptors, not from trying to force routing through all 512 experts per layer in a synthetic warmup.

The current implementation already provides:
- a deterministic writer
- a streaming payload writer with bounded host memory
- a streaming cache loader with bounded host memory
- `SingleTokenForwardModel::CreateFromCache(...)`

So the remaining work is payload redesign and validation, not inventing the cache subsystem from scratch.

## Implementation Steps

### Step 1: Define cache format and save path

- `model_cache.h`: header struct, tensor entry struct, read/write declarations
- Save after construction + warmup, gated by `NEMOTRON_MODEL_CACHE_DIR`
- Build the cache directly from the manifest and descriptor catalogs
- Make heuristic-cache serialization optional in v1
- Status:
  - implemented
  - first v1 payload currently stores execution-ready dense/scaled-FP8/embedding surfaces and aligned NVFP4 payloads
  - next iteration should switch dense/scaled-FP8/embedding entries back to packed checkpoint bytes

### Step 2: Implement cache loading

- `SingleTokenForwardModel::CreateFromCache(path)` alongside existing `Create()`
- One cudaMalloc, streaming file read, device-side conversion, view construction
- Populate heuristic cache from stored entries if present
- All experts immediately available — no lazy materialization on first forward call
- Status:
  - implemented
  - first execution fix landed after the initial run: cached Mamba tensors must be reconstructed on the flattened execution surface used by the runtime kernels

### Step 3: Validate

- Oracle regression with cache-loaded model (59/59)
- Decode oracle top-k agreement
- Cache vs uncached construction: bitwise weight comparison
- Cache invalidation on checkpoint hash change
- Measure: cache-load wall time, first-token time (should eliminate the 11.9s lazy cost)
- Status:
  - partially implemented
  - measured cache-backed decode now runs and predicts the correct token
  - current blocker is not correctness but TTFT: `~113.7s` cache-backed startup is much worse than the uncached `~27.9s` cold path

### Step 4: UMA and I/O optimizations

- Test `O_DIRECT` reads into `cudaMalloc`'d UMA memory
- Test `io_uring` for async reads if `O_DIRECT` works
- Page-align all cache payload offsets

## What Exists Already

- `DeviceNvfp4Weight::CreateView` — non-owning NVFP4 weight views, 256-byte aligned
- `DeviceDenseWeightFp32::CreateView` — non-owning dense weight views
- `UploadedLinearOp::CreateNvfp4View` / `CreateDenseView` — linear op wrappers
- `Nvfp4AlignedBuffers` / `UploadNvfp4AlignedBuffers` / `MaterializeNvfp4AlignedViewOp` — aligned lazy expert staging (reusable for cache views)
- `storage_conversion.cu` — device-side BF16→FP32, FP8→FP32 conversion
- `SingleTokenForwardBuildReport` — layer ordering and timing
- `GemmHeuristicCache` with `algorithm_ids_by_key_` map — technically serializable, but currently cheap enough that v1 cache support should treat it as optional

## Expected Result

| Path | model_build | first_token | end-to-end |
|---|---:|---:|---:|
| Current (lazy experts) | ~15s | ~12s | ~28s |
| Cache load (basic, ~67 GB at 6 GB/s) | ~11s | <1s | ~13s |
| Cache load (UMA direct, O_DIRECT) | ~11s | <1s | ~13s |
| Cache load (faster NVMe ~12 GB/s) | ~6s | <1s | ~8s |

The key improvement beyond raw construction time: **first-token decode drops from ~12s to <1s** because all experts are pre-resident. No lazy materialization, no first-use alignment staging, no on-demand cublasLt heuristic search.

Current measured reality for the implemented execution-ready v1:
- cache write:
  - payload `102103740420` bytes
  - `41643` entries
  - several minutes to generate on DGX Spark
- cache load:
  - `cache_load_ms ≈ 110.4s`
  - first decode `≈1.09s`
  - second decode `≈0.82s`

So the next focus is clear:
1. keep the streaming writer/loader and cache-backed constructor
2. redesign the dense/scaled-FP8/embedding payload back toward packed checkpoint bytes
3. rerun the same benchmark path after that redesign before treating model cache as the active TTFT path

Update after the packed-linear throughput pass:

- the native packed BF16 / FP8 execution work lowered uncached cold-start TTFT enough that model cache no longer needs to rescue the basic first-token path
- the aligned uncached path is now about:
  - `environment_build_ms ≈ 2103`
  - `model_build_ms ≈ 9603`
  - first decode `≈ 5064`
  - end-to-end first token `≈ 16.8s`
- warmed decode is still about `0.73s/token`, so the main remaining throughput issue is hot-path execution, not constructor cost

That changes the cache priority slightly:

- compact cache design is still valuable for TTFT
- but it should now be treated as an additive TTFT reduction on top of a much better uncached path, not as the primary fix for a broken constructor path
- the cache payload redesign should stay aligned with the throughput work:
  - keep dense / scaled-FP8 / embedding tensors packed
  - avoid reintroducing large FP32 execution surfaces
  - prefer precomputed metadata only where reconstruction is actually expensive (NVFP4 is still the strongest case)

## Scope Boundary

This plan covers weight caching only. It does not cover:
- Request-local state (KV pages, Mamba state) — per-request, not cacheable
- Execution graph capture — separate optimization, orthogonal
- Prefix cache warmup — separate from weight caching
