# Intentional Divergences

- Exact-prefix whole-state cache: shared reuse is modeled as exact token-prefix matches at explicit boundaries only: global roots and per-conversation committed heads. Each admitted node stores copied attention KV plus full Mamba conv and SSM state, then restores that whole state before tail prefill. This remains the primary architectural divergence from vLLM's finer-grained serving cache model.
- Single-process runtime boundary: this branch aligns execution semantics and token outputs with the local vLLM reference, but it keeps a single-process in-process C++ runtime. It does not attempt to mirror vLLM's broader multiprocessing serving control plane.
