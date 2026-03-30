# Initial Forward Pass Plan

Execution mode for this sub-plan: proceed continuously through the sequence unless a real blocker is encountered.

This note narrows Milestone 3 into a concrete implementation sequence for the first end-to-end, single-request forward path that can be checked against offline oracle outputs.

It is intentionally narrower than the full runtime plan:

- one request at a time
- no batching
- no prefix-cache restore/publish on the hot path
- no graph capture
- no speculative decode
- no MTP
- correctness first, then optimization

## Why This Sub-Plan Exists

The top-level plan already defines the right milestone, but Milestone 3 is still too broad to execute directly. At this point we have enough substrate to stop broad platform investigation and start wiring the actual model path:

- manifest-backed startup and weight ownership
- executable dense FP32 GEMM
- executable NVFP4 GEMM with checkpoint-backed oracle coverage
- executable BF16 paged attention
- Mamba update oracle fixtures and GB10 cache-format conclusions

What is still missing is the integration sequence that gets us from those isolated primitives to a real model forward pass with oracle checkpoints.

## First Success Target

The first success target is:

1. load the packed model manifest
2. build a single-request runtime context
3. run prompt prefill for one short deterministic prompt
4. produce final logits
5. compare final logits and selected intermediate states against an offline oracle

Immediately after that, extend the same path to:

1. one-token decode after prefill
2. request-local KV reuse
3. request-local Mamba state reuse

The initial target is not “fast serving.” It is “correct full-path execution.”

## Non-Goals For This Stage

Do not mix these into the first oracle-checked path:

- prefix-cache hits across requests
- scheduler/batching work
- speculative decoding
- MTP heads
- aggressive mixed-precision tuning beyond already-validated operator paths
- production memory-optimization work that obscures correctness

## Existing Building Blocks

Already available and should be reused directly:

- manifest and artifact loading
- startup-owned weight arena and catalogs
- uploaded FP32 dense weights and executable FP32 GEMM
- uploaded NVFP4 weights and executable NVFP4 GEMM
- BF16 cuDNN FE paged attention with KV-page and page-table scaffolding
- embedding upload and lookup
- Mamba one-step oracle fixtures
- Mamba target-trace oracle fixtures

That means the missing work is mainly integration, model metadata, request context, and oracle checkpoints for the composed path.

## Required New Runtime Surfaces

Before the full loop can exist, we need a few explicit runtime surfaces:

### 1. Model Schedule / Layer Registry

Add a runtime-owned layer schedule that answers:

- total layer count and per-layer family
- tensor names/bindings for that layer
- norm/gate/projection tensor bindings
- whether the layer uses attention, Mamba, routed experts, shared experts, or a combination

This should come from manifest-backed startup metadata, not from hard-coded assumptions scattered across kernels.

Exit criteria:

- a test can enumerate the full ordered layer schedule
- counts match the intended model structure
- required tensors for each layer family are validated at startup

### 2. Request Execution Context

Add a narrow request-owned execution context that owns:

- current hidden-state buffers
- scratch buffers
- per-request KV handles/pages
- per-request Mamba recurrent state
- current sequence lengths and decode position

This should stay concrete and model-specific enough to move quickly. Do not build a generic tensor runtime here.

Exit criteria:

- request context can be created, reset, and advanced
- KV and Mamba state are request-local and explicit
- tests cover size/layout sanity and reset semantics

### 3. Missing Primitive Ops

Add the minimal primitive set needed to compose real layers:

- residual add
- RMSNorm / layer norm as required by the checkpoint layout
- activation/gating ops required by the expert and Mamba paths
- router/top-k helpers required for routed experts
- any lightweight reshape/view helpers needed between existing operator calls

These should be added as small concrete operators, not as a generic graph engine.

Exit criteria:

- each primitive has at least one direct correctness test
- primitive outputs can be checked against CPU references or offline oracle dumps

## Forward-Pass Integration Sequence

### Stage A. Attention-Only Layer Slice

Build the first real layer slice around one attention layer:

- input norm
- Q/K/V projections
- positional handling exactly as required by the checkpoint
- paged attention call
- output projection
- residual merge

Use short deterministic prompts and compare:

- norm output
- Q/K/V or attention output at one checkpoint
- final layer output

Why first:

- the backend path is already selected and executable
- it validates request-local KV state handling
- it is the shortest path from isolated primitives to a real layer

Current status:

- a first real runtime slice exists
- a checkpoint-derived layer-7 oracle now gates final output and paged KV contents
- the current slice still uses correctness-first host-assisted BF16 staging, which remains an explicit optimization target later

### Stage B. Mamba Layer Slice

Build the first real Mamba slice using:

- required input projections
- conv / recurrent update path
- FP32 recurrent-state baseline
- output projection / residual merge

For the first pass, use FP32 recurrent state even if later policy prefers FP16+SR. The optimization decision is already known; correctness should still start from the clearest baseline.

Current prerequisites now in place:

- request-local recurrent-state storage
- request-local conv-state storage
- correctness-first scaled-FP8 linear support for `in_proj` / `out_proj`

Compare against oracle checkpoints for:

- selected intermediate projections
- next recurrent state
- layer output

Current status:

- a first real layer-0 decode-step Mamba slice now exists
- it covers:
  - outer block RMSNorm
  - scaled-FP8 `in_proj`
  - request-local conv-state update
  - request-local FP32 SSM-state update
  - grouped gated RMSNorm inside the mixer
  - scaled-FP8 `out_proj`
  - residual merge
- a checkpoint-derived layer-0 oracle now gates:
  - final layer output
  - updated conv state
  - updated SSM state
- the current slice is correctness-first and host-assisted for conv/state math; optimization comes later

### Stage C. Expert Slice

Build the first MoE slice using the already-validated GEMM paths:

- router logits
- top-k expert choice
- token-to-expert dispatch
- routed expert up/down path
- shared expert path
- merge back to hidden state

Start with correctness-first dispatch even if it is not the final fastest design.

Compare against oracle checkpoints for:

- router logits / selected experts
- routed expert output
- shared expert output
- merged layer output

Current status:

- a first real layer-1 expert slice now exists
- it covers:
  - outer block RMSNorm
  - dense router logits
  - correctness-first grouped top-k selection
  - scaled-FP8 `fc1_latent_proj`
  - routed expert up/down execution under the validated NVFP4 host pack/dequant contract
  - dense `fc2_latent_proj`
  - scaled-FP8 shared `up_proj`
  - shared NVFP4 `down_proj`
  - residual merge
- a checkpoint-derived layer-1 oracle now gates:
  - router logits
  - selected expert indices and weights
  - routed latent output
  - shared output
  - mixer output
  - final layer output
- the current slice is correctness-first and still host-assisted for routed and shared NVFP4 expert matmuls; optimization comes later

### Stage D. Full Layer Loop

Once the three family slices are correct in isolation:

- wire the ordered layer loop
- run prefill end to end
- add final norm / logits projection
- compare final logits against the oracle

Current blocker status:

- the dense BF16 checkpoint-ingestion blocker is now reduced:
  - the current dense upload path accepts BF16-backed descriptors and converts them into the existing FP32 execution surface
- a first registry-backed single-token full-loop scaffold now exists:
  - centralized known-model config
  - tested ordered-layer plan construction
  - request-local KV and Mamba state sizing
  - runtime-side single-token runner scaffolding around embeddings, per-layer dispatch, optional final norm, and logits projection
- actual single-token full-model runtime execution is no longer blocked on artifact availability:
  - the manifest-backed `full_forward_manifest_smoke_test` now executes the real `88`-layer checkpoint end to end
  - the current path uses correctness-first lazy weight materialization and CPU fallbacks for several dense BF16 / FP8 projections
- the larger remaining blocker for Stages 8 and 9 is now a practical oracle gate rather than artifact availability:
  - `RunPrefill(...)` exists and the Mamba slice now supports multi-row replay over recurrent state
  - but the full offline oracle path is still too slow for routine full-`88`-layer gating on short prompts
  - even a checkpoint-derived truncated prefix-prefill oracle through layer 7 is currently blocked by the Python MoE / NVFP4 reference cost at the first expert layer
- the first naive offline single-token full-model oracle generator now exists, but its current pure-Python MoE path is too slow for routine full `88`-layer generation

The first composed full loop should support:

- prefill only
- then prefill + one-token decode

Do not add sampling complexity before logits are trusted.

## Oracle Plan

We need a dedicated offline oracle for the first full path.

### Oracle Outputs To Dump

For one short deterministic prompt, dump:

- token IDs after serialization/tokenization
- embedding output
- selected per-layer checkpoints:
  - one attention layer output
  - one Mamba layer output
  - one MoE layer output
  - selected recurrent states
- final hidden state
- final logits
- one-token decode logits after prefill

The first oracle prompt should be short enough to debug comfortably, then expand to a slightly richer chat-shaped prompt.

### Oracle Gating Order

Do not jump straight to only final logits. Gate in this order:

1. embedding output
2. one attention slice
3. one Mamba slice
4. one expert slice
5. full prefill logits
6. one-token decode logits

That keeps divergence localized.

## Recommended Implementation Order

1. Add the model schedule / layer registry.
2. Add the request execution context.
3. Add missing primitive ops.
4. Build the attention-only layer slice and its oracle test.
5. Build the Mamba layer slice and its oracle test.
   Note: the remaining missing work is now the slice itself, not the basic FP8 or state-buffer plumbing.
6. Build the expert slice and its oracle test.
7. Build the full layer loop.
8. Add full prefill oracle comparison.
9. Add one-token decode oracle comparison.

Current reality:

- Step 7 now has a real manifest-backed runtime path for the single-token case, not only a scaffold.
- Step 8 now has a real narrowed gate:
  - a `4`-token `short_chat` prefix-prefill oracle that stops after layer `7`
  - the composed runtime can execute it
  - this gate is now green after fixing the oracle-side TF32 measurement artifact
- Step 9 now has a real all-layer gate:
  - the single-token decode oracle runs end to end through all `88` layers
  - the remaining failure has moved late, to the full decode path rather than the early composed prefill path
- The runtime now owns static forward-pass setup inside the model itself:
  - uploaded embedding table
  - final norm weight
  - LM head op
  - prepared attention / Mamba / expert slices
- The decode oracle now has practical localization controls:
  - `NEMOTRON_SINGLE_TOKEN_DECODE_STOP_LAYER=<n>`
  - `NEMOTRON_SINGLE_TOKEN_DECODE_SWEEP_LAYERS=a,b,c`
  - captured-layer-only comparison when intentionally truncated

Current immediate next step:

1. The earlier layer-19 decode boundary is no longer the current blocker.
   - after tightening the correctness-first `scaled_fp8_linear` host accumulation, the rebuilt decode sweep now stays green through `stop=23`
   - the first hidden-output gate failure moved to layer `24`
2. That change matters:
   - at least part of the earlier all-layer decode drift was not just “unavoidable NVFP4 amplification”
   - a real runtime-side linear-op accuracy improvement bought measurable end-to-end decode headroom
3. The layer-19 analysis remains useful context, but it is no longer the immediate target:
   - layer `19` is still amplification, not a split implementation path
   - the all-layer blocker is now the `24-31` region
4. The next task is therefore to rebuild the same exact-input replay workflow at the new boundary:
   - dump the rebuilt decode runtime state at `stop=23`
   - generate exact-input layer-24 fixtures and the immediate neighboring slice if needed
   - determine whether layer `24` is another scaled-FP8 accumulation-sensitive producer or a different composed decode mismatch
5. Only after that new boundary is localized should Step 9 move again.

Latest decode-localization update:

- layer `7` attention is now clean on exact dumped runtime input
- layer `6` Mamba is also clean on exact dumped runtime input
- the first failing standalone runtime-input boundary had been the layer `5` expert replay
- that layer-5 investigation is now much tighter:
  - FP8 activation round-trip matches the oracle exactly
  - FP8 weight dequant matches the oracle exactly
  - the remaining `fc1_latent` delta came from the host float32 accumulation surface in the correctness-first `scaled_fp8_linear` path
- after changing that host path to higher-accuracy accumulation:
  - the layer-5 exact-input expert replay now passes
  - `fc1_latent_diff` dropped from `2.26498e-06` to `3.44589e-07`
  - the routed-path mismatch collapsed from milliscale to `~1e-7`
- the rebuilt all-layer decode sweep now reflects that improvement:
  - `stop=19`: green
  - `stop=23`: green
  - first failing layer: `24`
  - by `stop=31`, the worst hidden-output drift is layer `30`

Current verification state:

- everything except the still-known all-layer single-token decode oracle blocker is green
- current broad regression count excluding that blocker is `58/58`

Latest decode-localization update:

- The first hidden-output gate failure is still layer `26` after the two host-accumulation fixes.
- But the exact-input replay story is now much cleaner than before:
  - layers `24` through `31` all replay cleanly on dumped runtime input
  - that includes alternating Mamba, attention, and MoE slices across the whole current failure band
- This means the current blocker is no longer “the layer-24/25/26 slice is broken”.
- The stronger current interpretation is:
  - some earlier decode drift was reducible implementation error, and two such sources have already been removed
  - the remaining `24-31` failure band currently looks like chained amplification of smaller upstream differences, not a newly isolated local slice bug in that band
- So the next debugging pass should stay disciplined:
  - keep walking forward only until the next standalone exact-input replay actually fails
  - if no such failure appears soon, shift the decode gate discussion toward final-logit and token agreement, because hidden-state-only tolerances may now be stricter than the corrected runtime path can realistically satisfy under the checkpoint's mixed-precision contract
- The first full functional check now supports that shift:
  - final decode top-1 token agrees exactly with the oracle
  - top-5 overlap is `5/5`
  - top-10 overlap is `10/10`
  - top-20 overlap is `17/20`
  - top-50 overlap is `45/50`
  - even though hidden-state/logit `rel_l2` is still too high for the current end-to-end gate
- That means Step 9 should split into two separate notions of correctness:
  - strict hidden-state/logit localization gates for debugging
  - functional decode gates for end-to-end serving correctness
- That split is now implemented:
  - truncated / stop-layer decode runs still use the strict hidden-state gate
  - the full all-layer decode oracle now passes or fails on functional decode behavior plus a final-logit `rel_l2` tripwire
- Current Stage 9 status:
  - the all-layer single-token decode oracle is now green on the real manifest
  - the remaining hidden-state drift is still visible and still useful for future precision/debug work
  - but it is no longer blocking the initial full-forward milestone

## Definition Of Done For This Sub-Plan

This sub-plan is complete when all of the following are true:

- a single request can run through the full model forward path
- short deterministic prompts produce stable final logits
- one-token decode after prefill also works
- selected intermediate states match offline oracle checkpoints closely enough to localize regressions
- the path still runs with:
  - no batching
  - no prefix cache
  - no graph capture
  - no speculative decode

At that point the runtime is ready to resume the main plan with:

- request-local state reuse and cache wiring
- batching/scheduler work
- further mixed-precision rollout where correctness is already anchored
