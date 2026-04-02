# Plan: Production Decode Attention Kernel

Project directory: `./proj-2026-04-02-0800`

## Goal

Replace the scalar attention fallback (1 thread per query head, shared memory scales with KV length) with a production decode attention kernel that eliminates KV-length-dependent shared memory, dramatically improves GPU utilization, and flattens the decode latency curve to enable consistent ≤20ms/token.

Current: 21.9ms/token mean, 19.4ms min, 24.4ms max. Early decode steps hit 20ms; later steps degrade because the scalar attention kernel scales linearly with sequence position. Attention is 24% of GPU time (52ms total / 17 tokens = ~3ms/token average, but scaling from ~0.5ms at start to ~1.5ms at position 32).

Nano attention config: `head_dim=128`, `query_head_count=32`, `kv_head_count=2`, GQA ratio 16:1, `tokens_per_page=16`.

## Current Scalar Fallback (what we're replacing)

`PagedAttentionDeviceFallbackKernel` in `attention_device_fallback.cu`:
- `kAttentionFallbackBlockSize = 1` — one thread per block
- One block per (batch, query_head, query_token) vector
- Single thread does: serial QK dot product over head_dim=128, serial softmax over all visible KV tokens, serial value accumulation
- `extern __shared__ float score_scratch[]` sized to `visible_kv_tokens` — scales with sequence length
- BF16 input/output, FP32 accumulation

## vLLM/TRT-LLM Reference

- vLLM `chunked_prefill_paged_decode.py`: separate decode kernel from prefill
- TRT-LLM `fmhaRunnerParams.h`: distinct kernel types for context vs generation
- FlashInfer: paged decode attention with online softmax, no sequence-length shared memory
- Key design: one warp per query head, threads cooperate on KV sequence reduction, online softmax (running max + running sum), tiled value accumulation

## Steps

- [x] **1. Add production paged decode attention kernel**
  Add `RunPagedAttentionDecodeProduction()` alongside the existing fallback. Design based on Codex review corrections:
  - **One block per KV head** with multiple warps handling the query group (16 query heads per KV head for Nano)
  - Grid: `batch_size * kv_head_count` blocks (for Nano decode: 1 × 2 = 2 blocks)
  - Block size: 256 threads (8 warps). Each warp handles 2 query heads from the group of 16.
  - **Head_dim sharded across lanes**: each lane in a 32-thread warp owns `head_dim / 32 = 4` dimensions of the query, QK accumulation, and output. This keeps register usage at ~8 floats/lane (4 query + 4 output) instead of 256.
  - For each KV token in the sequence:
    - All warps in the block load the same K vector (from shared KV head), sharded across lanes
    - Each warp computes QK dot product for its 2 query heads using warp-level reduction
    - Online softmax update: each warp maintains `(max_score, running_sum, partial_output[4])` per query head — O(head_dim/32) state, NOT O(kv_length)
    - Load V vector, accumulate weighted value into partial output
  - After all KV tokens: each warp writes its 2 query heads' outputs
  - GQA mapping: `kv_head = query_head / (query_head_count / kv_head_count)` — query heads 0-15 share KV head 0, heads 16-31 share KV head 1
  - **No shared memory scaling with KV length.** Fixed smem for K/V tile loading only.
  - Compute still scales linearly with KV length (unavoidable for exact attention), but memory footprint is O(1) and GPU utilization is much higher.
  Key files: `runtime/src/backend/attention_device_fallback.cu`, `runtime/include/nemotron/attention_device_fallback.h`

- [~] **2. Wire production kernel into attention layer dispatcher**
  In `attention_layer.cpp`, add an explicit dispatcher:
  - `token_count == 1` AND not cuDNN → use production decode kernel
  - `token_count > 1` → use existing fallback (or cuDNN if available)
  - `NEMOTRON_FORWARD_ATTENTION_SCALAR_FALLBACK=1` → force old fallback
  Keep the existing device-vs-host compare hook behind `NEMOTRON_FORWARD_COMPARE_DEVICE_ATTENTION`.
  Key files: `runtime/src/backend/attention_layer.cpp`

- [ ] **3. Verify correctness and benchmark**
  - Smoke test PASS
  - Benchmark steady-state decode: target ≤20ms/token mean
  - Verify attention time is now ~constant across decode positions (not scaling with KV length)
  Key files: benchmark scripts

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Warp-cooperative decode attention kernel | done | — | 2 blocks × 256 threads, GQA-aware, online softmax, lane-sharded; compile PASS |
| 2 | Wire into attention layer dispatcher | done | — | NEMOTRON_FORWARD_ATTENTION_PRODUCTION gate; smoke PASS |
| 3 | Verify and benchmark | done | — | 13.8ms/token, 72.5 tok/sec, 1657x from baseline |
