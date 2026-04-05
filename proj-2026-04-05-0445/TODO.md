# Project TODO

- The post-attention rewrite is complete. The current next-tier priorities are:
  - runtime FP32→NVFP4 routed-input packing/scaling
  - reduced-launch grouped routed-expert execution
  - Mamba prefill optimization
