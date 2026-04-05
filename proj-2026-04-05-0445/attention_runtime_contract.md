# Attention Runtime Contract

This file freezes the runtime-facing contract for the new multi-token attention
backend chosen in `attention_architecture_decision.md`.

## Scope

This contract applies only to the fast path for:

- RTX 5090
- consumer `SM120`
- Nemotron 3 Nano NVFP4
- BF16 paged KV cache
- single GPU
- low concurrency (`<=4`)
- exact-prefix cache restore flow already used by this runtime

## KV / Page ABI

The new backend must preserve the current paged-KV ABI:

- KV cache dtype: BF16 only
- KV cache tensor layout:
  - key cache: `[attention_total_pages, kv_heads, tokens_per_page, head_dim]`
  - value cache: `[attention_total_pages, kv_heads, tokens_per_page, head_dim]`
- page size: `tokens_per_page = 16`
- page-table type: `int32`
- batch plan shape:
  - `PagedAttentionBatchPlan.page_table`
  - `PagedAttentionBatchPlan.sequence_lengths`
  - `PagedAttentionBatchPlan.pages_per_sequence`
  - `PagedAttentionBatchPlan.max_pages_per_sequence`

Do not invent a second KV/page ABI for the fast path.

## Geometry Contract

- query heads: `32`
- KV heads: `2`
- head dim: `128`
- causal only
- retained context target: up to `64k`
- batch size target: `1..4`

Any configuration outside this contract is allowed to use a fallback path.

## Backend Split

The production backend policy is:

- `token_count == 1`
  - keep the existing decode-specialized path only if it still wins after the
    new multi-token backend exists
- `2 <= token_count <= 1024`
  - use the new multi-token attention backend
- `token_count > 1024`
  - decompose into bounded multi-token chunks and run the new backend per chunk
- unsupported / debug / stats-only cases
  - fall back to the existing generic path

This project does **not** target multiple competing production multi-token
backends.

## Query Chunking Policy

The chunking policy is fixed now:

- `max_multi_token_query_tokens = 1024`
- large prefills and large uncached tails are decomposed into contiguous query
  chunks of at most `1024` tokens
- smaller tails may use any `2..1024` query length directly; they do not need a
  separate chunk-family-specific kernel contract

Implications:

- `q256` runs as one `256`-token launch
- `q1k` runs as one `1024`-token launch
- `q4k` runs as four `1024`-token launches
- long-turn extends such as `q128s64k` run as one `128`-token launch

This keeps the policy simple while still matching the benchmark buckets used in
the oracle runs.

## Prefix-cache Interaction

The exact-prefix cache model remains unchanged:

- restore exact request state first
- process only the uncached tail
- advance the live sequence state through the same request-owned KV pages

The new backend must fit this flow. It is not allowed to assume a global shared
block cache model like vLLM's serving path.

## Fallback Rules

The new fast path is not required to handle:

- non-BF16 KV cache
- different page sizes
- different Nano-incompatible head geometry
- DCP / EP / TP
- sink attention
- multimodal attention

Those cases may remain on the existing fallback path.

## Scheduler / Concurrency Policy

- optimize batch size `1` first
- verify behavior at batch size `4`
- long concurrent prefills may be serialized by higher-level policy rather than
  forcing the attention backend to optimize every pathological case equally

This contract is for the common local serving path, not a general multi-tenant
server design.

## Immediate Implementation Consequence

The next implementation step should add one new multi-token backend to
`AttentionBackend`, wire it into `AttentionBackendPolicy`, and keep the current
device fallback only as a correctness backstop.
