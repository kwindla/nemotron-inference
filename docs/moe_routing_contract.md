# MoE Routing Contract

## Canonical Tensors

The canonical routed-expert contract is the token-major `topk_ids` / `topk_weights`
pair emitted by grouped top-k selection and consumed by MoE execution:

- `topk_ids: int32[token_count, top_k]`
- `topk_weights: float32[token_count, top_k]`

The first dimension is always token-major. Row `t` contains the `top_k` routed
experts for token `t`, and slot `s` in `topk_weights[t, s]` always corresponds
to `topk_ids[t, s]`.

## Selection Semantics

The selection contract matches vLLM's `grouped_topk()` semantics in
`third_party/vllm/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`.

- Router logits are activated with sigmoid for Nemotron Nano style routing.
- `e_score_correction_bias` affects expert selection only. It does not change the
  routing weight gathered for a selected expert.
- Expert ordering is deterministic and batch-invariant. Within a token row,
  experts are ordered by descending selection score, with ties broken by lower
  expert index first. This is the contract-equivalent of vLLM's `sorted=True`
  grouped-top-k behavior.
- The output tensors preserve that sorted order exactly. Consumers must not
  reshuffle slots unless they are explicitly adapting into a different internal
  representation.

For Nano's current configuration (`n_group = 1`, `topk_group = 1`), grouped
selection degenerates to a single sorted top-k over the full expert set, but
the canonical contract is still defined in grouped-top-k terms so the same
ordering rules remain valid if grouped routing is re-enabled.

## Weight Semantics

`topk_weights` carry routing weights, not selection scores.

- Start from the unbiased activated score for each expert.
- Use the bias-adjusted score only to decide which experts are selected.
- Gather the unbiased activated scores for the selected experts.
- If `norm_topk_prob == true`, renormalize each token row so the selected
  weights sum to `1.0` before scaling.
- Apply `routed_scaling_factor` after optional renormalization.

That means:

- `norm_topk_prob == true`: `topk_weights[t, :]` sums to
  `routed_scaling_factor`.
- `norm_topk_prob == false`: `topk_weights[t, :]` is the raw activated score
  gather multiplied by `routed_scaling_factor`.

## Contract Boundary

The canonical API boundary ends at token-major `topk_ids` / `topk_weights`.

The following expert-major tensors are adapter-owned, not contract-owned:

- `expert_offsets`
- `sorted_token_indices`
- `sorted_token_weights`
- `active_expert_count`
- `active_expert_ids`

Those buffers are valid adapter surfaces for kernels that need expert-major
compaction, but they are downstream of the canonical contract. They must be
derived from the canonical token-major tensors without changing slot order or
weight semantics.

## Host/Device Boundary

Fast-path routing is device-native:

- `RunDeviceExpertSelection()` defines the canonical contract.
- `RunDeviceExpertRouting()` is a device-side compaction adapter.

Host-side reconstruction of expert-major routing tables is permitted only in
explicit debug, fallback, or backend-adapter paths. It is not part of the
canonical MoE routing contract, and fast-path regressions should fail if the
runtime starts rebuilding canonical routing state on the host by default.
