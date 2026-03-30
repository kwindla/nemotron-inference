# V1 Cache Architecture

## Goal

Minimize TTFT for the actual target workload:

- append-only multi-turn chat
- long shared system instructions
- low concurrency, up to 8 active requests
- target context length up to 64k

The design target is **true reusable prefix state**, meaning both:

- attention KV
- full current Mamba recurrent state

For v1, the cache should optimize for the boundaries that matter most instead of snapshotting densely at small token intervals.

## Core Decision

V1 uses **exact-prefix cache nodes** with full reusable state at selected boundaries:

- global shared roots
- per-conversation committed heads
- optional per-conversation prompt heads

V1 does **not** use dense small-block Mamba snapshotting as the primary design.

## Request Model

The runtime should continue to accept a full chat-completions style request payload.

Optional non-standard extension:

- `conversation_id`

`conversation_id` is a performance hint, not a correctness dependency.

Correctness rules:

- the full prompt is always present in the request
- cache reuse is validated against exact token IDs after serialization/tokenization
- if the cache misses, the request still runs correctly from scratch

## Cache Node Classes

### Global Shared Root

Represents a hot prefix reused across unrelated requests, such as:

- long system instructions
- tool schemas
- house prompt boilerplate

### Conversation Committed Head

Represents the end of the last completed assistant turn for a conversation.

This is the primary v1 fast path for multi-turn chat.

### Conversation Prompt Head

Represents the end of the current user turn before assistant generation.

This is optional in v1 and mainly exists to support:

- regenerate
- retry with identical prompt state

## Node Contents

Each reusable node stores:

- exact token IDs for the cached boundary
- token-count metadata
- fast hash for quick reject
- namespace / tenant metadata
- serializer and tokenizer revision metadata
- model revision metadata
- attention KV state handles
- full current Mamba recurrent state handles
- byte accounting
- allocator-owned reusable-state descriptors with deterministic retain/release semantics on publish, replacement, and eviction
- LRU metadata

If the runtime materializes Mamba recurrent state as multiple tensors, the reusable state must include the full set required to resume generation correctly. In the current reference implementation this includes both:

- conv state
- SSM state

## Lookup Flow

1. Serialize and tokenize the full request.
2. If `conversation_id` is present, probe that conversation's committed head first.
3. If the committed head exists and its tokens are an exact prefix of the request tokens, use it.
4. Otherwise, search the global prefix index for the longest exact-prefix match.
5. Optionally, if prompt-head mode is enabled and the request is a regenerate/retry, check the conversation prompt head.
6. Restore cached KV and Mamba state from the best matching node.
7. Prefill only the uncached tail.
8. Decode the assistant response.
9. On successful completion, commit a new conversation head node.

## Commit Rules

After the assistant turn completes:

- create or refresh the conversation committed-head node at the new prefix boundary
- update the conversation map to point to that node
- keep only one committed head per conversation in v1

If prompt-head mode is enabled:

- materialize a prompt-head node after user-turn prefill
- use it for regenerate or retry on the same turn
- do not treat prompt heads as conversation history

## Runtime Control

V1 should expose a full prefix-cache disable switch through runtime configuration, for example:

- environment variable such as `NEMOTRON_PREFIX_CACHE=0`
- CLI/config equivalent if the process does not rely on env vars

When disabled:

- bypass cache lookup
- bypass cache admission and publication
- zero out effective shared-cache reservation during startup
- run the uncached path correctly
- allow clean benchmark/control comparisons against the cached path

## Streaming Rule

Do not publish partial streamed assistant output into the committed conversation head unless that exact partial output is intended to become the conversation history visible to the client.

## Concurrency Rule

For v1:

- allow at most one mutating in-flight request per conversation
- reject or serialize concurrent advances of the same conversation

This keeps cache-head semantics simple and avoids ambiguous write ordering.

## Global Promotion

V1 should support promotion of hot shared roots into the global cache.

Reasonable promotion signals:

- exact prefix reused by multiple conversations
- long prefix length
- recent repeated hits

Global promotion must preserve exact token identity and namespace isolation.

## Eviction

Evict by **bytes**, not by entry count.

Recommended v1 policy:

- weighted LRU over reusable nodes
- per-namespace quotas
- explicit cleanup of any conversation-head mapping that points at an evicted node

Do not evict request-local speculative state into the shared cache. Shared cache entries must only contain fully committed reusable nodes.

## Why This Design Fits The Workload

For append-only chat, the most valuable reusable boundary is the full conversation prefix at the last completed assistant turn.

For shared system prompts, the most valuable reusable boundary is the shared root itself.

This means v1 can get most of the TTFT benefit without paying the memory cost of storing a full Mamba snapshot every small block.

## Deferred Work

These are later paths, not the v1 foundation:

- dense partial-prefix checkpoints
- hierarchical checkpoints with bounded recompute
- host spill for reusable nodes
- speculative-aware cache-node branching
- more advanced promotion heuristics

## Testing Requirements

V1 cache tests must cover:

- exact-prefix validation against token IDs
- conversation-ID fast path
- fallback to global longest-prefix lookup
- committed-head append-only reuse
- regenerate from prompt head
- eviction by bytes
- namespace isolation
- continuation parity after restoring cached state
