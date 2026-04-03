# vLLM Alignment Diff for Nemotron Nano NVFP4 on RTX 5090

## Baseline

- External reference implementation: `third_party/vllm` at `v0.19.0`
- Relevant model path: `vllm/model_executor/models/nemotron_h.py`
- Relevant MoE backend for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`: `flashinfer_cutlass`
- Not the right latency baseline for Nano: `flashinfer_trtllm`, because the Nano hidden size is `2688` and vLLM explicitly skips that path when `hidden_dim % 512 != 0`

For this hardware and model, the comparison target is not "any vLLM MoE path". It is the vLLM NVFP4 Blackwell path built around:

- GPU top-k routing
- load-time backend-native weight preprocessing
- one fused experts call per chunk
- the same fused MoE stack for both chunked prefill and normal execution

## Concrete Diff Against the Current Plan

### 1. Goal and success criteria

Current plan:

- Optimize a prefill-specific owned routed-expert kernel, with `64` tokens as the first design center
- Treat `32`-token TTFT as a regression/benchmark case
- Keep the current `token_count == 1` decode path unchanged

vLLM-aligned diff:

- The first-class goal should be a unified routed MoE execution surface that can win on both decode and prefill
- `64`-token chunking can remain an important tuning point, but it should not define a separate architecture for prefill only
- Success must include head-to-head comparison against vLLM `flashinfer_cutlass` on this exact model and hardware for both prefill and decode

Change to make in the plan:

- Replace "prefill kernel launch optimization" as the primary framing with "unified routed MoE backend alignment against the vLLM NVFP4 baseline"
- Treat prefill chunk size as a tuning knob, not the boundary between two different execution designs

### 2. Backend target

Current plan:

- Make an owned SM120-specific persistent routed-expert kernel the mainline optimization target

vLLM-aligned diff:

- vLLM reaches the relevant baseline through a backend abstraction and a fused experts implementation first
- The architectural win is not "persistent kernel" by itself; it is "GPU routing + prepared weights + fused expert execution"

Change to make in the plan:

- Make a vLLM-class fused backend interface the first milestone
- Move the owned SM120 kernel to a follow-on backend that is justified only if it beats the vLLM-class path in measurement

### 3. Routing contract

Current plan:

- Step 2 is built around explicit expert-major routing outputs: `expert_offsets`, `active_expert_ids`, sorted token indices/weights, and bucket metadata

vLLM-aligned diff:

- vLLM's first-class contract is token-major GPU top-k output: `topk_ids` and `topk_weights`
- The fused experts backend consumes those tensors directly
- `flashinfer_cutlass` does not require an expert-map style routing table

Change to make in the plan:

- Make `topk_ids` and `topk_weights` the first-class routed MoE interface
- Keep expert-major compaction only as an adapter for the legacy path or for a future custom backend that truly needs it
- Do not keep investing in expert-major routing as the center of the design if the chosen backend does not consume it

### 4. Weight layout and preprocessing

Current plan:

- Consume monolithic expert weights directly and avoid introducing a new weight layout

vLLM-aligned diff:

- vLLM converts NVFP4 MoE weights into backend-native kernel format after loading
- That includes layout conversion and scale shuffling so the runtime path does not pay format costs repeatedly

Change to make in the plan:

- Add load-time backend-native MoE weight preparation as a first-class milestone
- Keep raw monolithic weights only for fallback, validation, and debugging
- Stop treating "no new weight layout" as a design constraint for the fast path

### 5. Decode path

Current plan:

- Preserve the current decode-only fused kernel as an architectural constant

vLLM-aligned diff:

- vLLM uses the same fused MoE framework across prompt and decode traffic
- Backend selection is an implementation detail inside the MoE stack, not a separate model architecture

Change to make in the plan:

- Demote the current scalar/shared-memory decode kernel to "candidate backend" or "fallback backend"
- Keep it only if it still wins after the unified fused backend lands and is benchmarked

### 6. Shared experts

Current plan:

- Shared expert is a companion fused kernel after the routed path is stable

vLLM-aligned diff:

- Shared experts already live inside the same `SharedFusedMoE` layer surface
- Overlap is optional, but the execution interface is unified from the start

Change to make in the plan:

- Put routed and shared experts behind the same MoE execution surface from day one
- Treat overlap and multistream execution as a tuning phase, not as a separate architecture exercise

### 7. Chunking and capacity control

Current plan:

- Introduce `prefill_window_tokens` and later add model-level orchestration around it
- Keep the current request-capacity coupling until that lands

vLLM-aligned diff:

- MoE chunk size is already a runner-level concept through `max_num_tokens`
- Chunked and non-chunked execution share the same MoE implementation surface

Change to make in the plan:

- Introduce a MoE-specific "max tokens per MoE call" control now
- Drive chunking at the model/runner level instead of hiding it behind expert-layer fallback
- Keep request/KV capacity and MoE execution window as separate concepts everywhere

### 8. Benchmarking discipline

Current plan:

- Optimize against internal TTFT and internal expert-layer breakdowns

vLLM-aligned diff:

- Internal wins are necessary but not sufficient
- The external bar is the vLLM implementation for the same model, quantization, and GPU family

Change to make in the plan:

- Add direct external benchmark tracking versus vLLM `flashinfer_cutlass`
- Require both correctness and performance comparison for decode and prefill before committing to a custom-kernel branch

### 9. What not to optimize next

Do not spend the next milestone on:

- more host-path cleanup in the cuBLASLt fallback
- more expert-major routing machinery unless a chosen backend requires it
- locking the design around a prefill-only persistent kernel before the backend surface and weight-prep surface exist

## Recommended Plan

### Phase 0: Lock the external baseline

- Treat vLLM `flashinfer_cutlass` as the performance bar for Nemotron Nano NVFP4 on RTX 5090
- Add a local benchmark matrix that compares our runtime and vLLM for:
  - single-token decode
  - short tail prefill (`32`, `64`)
  - medium prefill (`128`, `256`)
- Record both end-to-end latency and MoE-layer time

### Phase 1: Introduce a unified MoE backend surface

Add an internal MoE backend interface with at least:

- `PrepareWeights(...)`
- `Run(...)` for `token_count >= 1`
- `Supports(config, token_count, device)`

Planned backends:

- legacy batched cuBLASLt path
- current decode scalar kernel
- unified fused backend
- optional future custom SM120 backend

The important change is that decode and prefill both route through the same selection and backend-selection surface.

### Phase 2: Make GPU top-k the primary routing product

- Keep router selection on device
- Make `topk_ids` and `topk_weights` the primary interface between routing and expert execution
- Keep expert-major routing compaction as a secondary adapter only

This aligns the runtime to the contract used by the relevant vLLM backend and avoids baking a custom expert-major layout into every future decision.

### Phase 3: Add load-time backend-native weight preparation

- Prepare NVFP4 MoE weights once after load into the format required by the chosen fused backend
- Store the prepared representation on the layer object and reuse it for both decode and prefill
- Keep raw weights only for fallback and validation

This is the highest-leverage structural change after GPU top-k because it removes layout friction from the hot path and makes a fused backend practical.

### Phase 4: Separate MoE execution window from request capacity

- Add a dedicated MoE chunk/window knob, for example `moe_max_tokens_per_call`
- Keep `max_tokens` for request capacity, KV planning, and request bookkeeping
- Drive long-prefill chunking in the model runner, not inside expert-layer fallback logic

The execution goal is:

- same MoE backend surface for unchunked and chunked traffic
- same MoE backend surface for decode and prefill

### Phase 5: Land a unified fused backend before a custom kernel

Preferred order:

1. Land a fused backend that matches the vLLM architecture: GPU top-k inputs, prepared weights, fused expert execution.
2. Benchmark it for both decode and prefill.
3. Compare it against the current scalar decode kernel and the current batched prefill path.

If direct FlashInfer/CUTLASS integration is acceptable in this codebase, that is the fastest path to a vLLM-class implementation. If it is not acceptable, the internal backend should still mirror that contract closely.

### Phase 6: Only then decide whether to keep or replace the current decode kernel

- If the current decode-only kernel still wins at `token_count == 1`, keep it as a specialized backend under the unified MoE surface
- If the unified fused backend matches or beats it, remove the architectural special-casing and keep the simpler design

The current decode kernel should not be preserved by policy. It should survive only by benchmark.

### Phase 7: Only then consider a custom SM120 backend

Pursue an owned SM120 routed-expert kernel only if all of the following are true:

- unified fused backend is correct
- load-time weight preparation is in place
- chunking is runner-level and shared across decode/prefill
- vLLM-class architecture still leaves measurable performance on the table

At that point, the custom backend can target the remaining gap instead of trying to solve routing, weight layout, chunking, and execution-shape problems all at once.

## Practical Priorities

If the goal is "fastest possible on this hardware for both decode and prefill", the next engineering order should be:

1. Benchmark vLLM locally and make it the explicit bar.
2. Refactor our MoE path around a unified backend surface.
3. Make GPU `topk_ids` / `topk_weights` the first-class routing output.
4. Add load-time NVFP4 backend-native weight preparation.
5. Land one unified fused backend and benchmark it for decode and prefill.
6. Keep or drop the current decode kernel based on measurement.
7. Build a custom SM120 backend only if it still has a clear measured upside.

That ordering matches the structure that already works in vLLM and minimizes the risk of spending weeks on a custom prefill kernel while remaining structurally behind on decode, chunking, and weight preparation.
