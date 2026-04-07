# Nemotron Runtime

From-scratch C++/CUDA inference implementations for Nemotron-3 hybrid Mamba-attention-MoE models, specialized per model and hardware platform.

Each platform specialization is a self-contained implementation optimized for its specific model variant, hardware target, and performance goals. Specializations do not share runtime code — they are free to diverge on data layouts, kernel contracts, execution paths, and precision policies.

## Platforms

| Directory | Model | Hardware | SM Arch |
|---|---|---|---|
| [platforms/super-spark](platforms/super-spark/) | Nemotron 3 Super NVFP4 | DGX Spark (GB10) | SM 121 |

## Quick Start

```bash
# Build a specific platform (default: super-spark)
cmake -S . -B build
cmake --build build -j2

# Or select explicitly
cmake -S . -B build -DNEMOTRON_PLATFORM=super-spark
```

See each platform's README for platform-specific build instructions, status, and test commands.

## Repo Layout

- `platforms/` — self-contained platform specializations (runtime, tests, benchmarks, docs, tools)
- `docs/` — cross-cutting research notes
- `server/` — multi-model serving infrastructure
