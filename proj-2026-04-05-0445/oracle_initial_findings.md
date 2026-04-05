# Initial vLLM Oracle Findings

Artifacts:

- `oracle_runs/20260405T090317Z` — smoke validation
- `oracle_runs/20260405T090413Z` — primary single-request matrix
- `oracle_runs/20260405T090507Z` — representative concurrency-4 matrix

## Environment

- vendored vLLM is being used:
  `/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/__init__.py`
- benchmark-resolved dtype is `torch.bfloat16`
- device capability is `SM120` (`12.0`)
- `current_platform_family100 = false`
- `supports_trtllm_attention = false`
- `flashinfer` imports successfully
- `triton` imports successfully
- Python `flash_attn` module is not present, but the vLLM `FLASH_ATTN` backend
  still benchmarks successfully in this environment

## Single-request results

### Prefill

- `q256`: `FLASH_ATTN` is best at `12.8 us/layer`
- `q1k`: `FLASH_ATTN` is best at `74.6 us/layer`
- `q4k`: `FLASHINFER` is best at `802.3 us/layer`, with `FLASH_ATTN` within
  about `1.3%`

### Extend

`FLASHINFER` is best on every single-request extend case in the frozen matrix:

- `q128s8k`: `96.4 us/layer`
- `q128s32k`: `341.4 us/layer`
- `q128s64k`: `676.4 us/layer`
- `q256s8k`: `183.3 us/layer`
- `q256s32k`: `679.0 us/layer`
- `q256s64k`: `1.342 ms/layer`
- `q512s8k`: `443.4 us/layer`
- `q512s32k`: `1.680 ms/layer`
- `q512s64k`: `3.338 ms/layer`
- `q1ks8k`: `823.8 us/layer`
- `q1ks32k`: `3.312 ms/layer`
- `q1ks64k`: `6.623 ms/layer`

`TRITON_ATTN` is consistently the slowest backend in these runs.

## Concurrency-4 results

### Prefill

- `4q256`: `FLASH_ATTN` is best at `28.7 us/layer`
- `4q1k`: `FLASHINFER` is best at `218.9 us/layer`

### Extend

`FLASHINFER` is best on every representative four-request extend case:

- `4q128s8k`: `437.2 us/layer`
- `4q256s32k`: `3.308 ms/layer`
- `4q1ks64k`: `22.20 ms/layer`

## Immediate implications

- The first external oracle signal does **not** support a TRTLLM-attention end
  state on this consumer `SM120` box. The benchmark probe says the relevant
  TRTLLM attention support path is unavailable here.
- The closest high-performing prior art for the long-context multi-token path
  on this machine is FlashInfer-like behavior, not the current runtime fallback
  and not TRTLLM attention.
- For tiny prefills, `FLASH_ATTN` remains competitive and sometimes best, so
  the architecture choice still needs to decide whether:
  - one backend should own both prefill and extend, or
  - a small split backend policy is justified on Nano / RTX 5090
- The architecture decision has now been taken in
  `attention_architecture_decision.md`. The next project step is to build the
  attention-only correctness and profiling harness for the chosen backend
  direction.
