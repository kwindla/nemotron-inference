# CUTLASS MoE Device Dispatch Design

Clean design for fully device-resident CUTLASS MoE expert dispatch. No host pointer arrays, no per-call cudaMalloc, no D→H copies.

## Problem With Current Approach

The current CUTLASS integration builds host-side pointer arrays per call (A, B, C, D, SFA, SFB), uploads them via DeviceBuffer::CopyFromHost, then passes them to the CUTLASS plan. This:
- Involves H→D copies every token
- Can race with async CUTLASS execution
- Is the pattern we're trying to eliminate
- Caused the full-model correctness bug (predicted_token_id=0)

## Clean Design

### Pre-Allocated Per-Layer State (created once during model construction)

For each MoE layer, store in `ExpertLayerSlice::Impl`:

```
cutlass_up_plan: CutlassNvfp4GroupedGemmPlan  // pre-built strides, layouts, workspace
cutlass_down_plan: CutlassNvfp4GroupedGemmPlan

// Device arrays sized for top_k groups, pre-allocated once:
cutlass_a_ptrs:     DeviceBuffer<const void*>  [top_k]  // A data pointers
cutlass_a_sf_ptrs:  DeviceBuffer<const void*>  [top_k]  // A scale pointers  
cutlass_c_ptrs:     DeviceBuffer<const void*>  [top_k]  // C pointers (output, beta=0 so unused)
cutlass_d_ptrs:     DeviceBuffer<void*>        [top_k]  // D pointers (output)

// For down_proj: aligned activation buffer, pre-allocated once:
cutlass_down_act_packed:  DeviceBuffer<uint8_t>  [top_k * aligned_packed_row]
cutlass_down_act_scales:  DeviceBuffer<uint8_t>  [top_k * aligned_scale_row]
```

### Per-Token Dispatch (all on device)

1. **Top-k selection** → produces `selected_indices_device` and `selected_weights_device` (already on device)

2. **Activation packing** → `PackDeviceRowMajorFp32ToNvfp4` produces packed latent + scales (already on device)

3. **Build A pointer arrays** (device kernel): all groups point to the same packed activation
   ```
   // Trivial fill kernel — all top_k entries point to the same buffer
   FillPointerArray<<<1, top_k>>>(cutlass_a_ptrs, latent_packed->packed_data(), top_k);
   FillPointerArray<<<1, top_k>>>(cutlass_a_sf_ptrs, latent_packed->block_scales_data(), top_k);
   ```

4. **Build B pointer arrays** (device kernel): gather from expert lookup table using selected indices
   ```
   // Existing GatherExpertSelectionLookupsChecked already does this,
   // producing up_weights_packed_ptrs_device and up_weights_raw_scale_ptrs_device
   ```

5. **Build D pointer arrays** (device kernel): each group writes to a different output row
   ```
   BuildStridedPointerArray<<<1, top_k>>>(
       cutlass_d_ptrs, grouped_up->data(),
       routed_expert_intermediate_size, top_k);
   ```

6. **CUTLASS up_proj** → `cutlass_up_plan->Run(a_ptrs, a_sf, b_ptrs, b_sf, c_ptrs, d_ptrs, 1.0, 0.0)`

7. **Post-scale** → `ScaleRowsByTensorScaleFp32(grouped_up, act_ts_device, weight_ts_device, top_k)`

8. **relu2 + NVFP4 repack** → existing `ScaleRelu2PackRowsToNvfp4`, but with aligned row stride for CUTLASS

9. **Build down A pointer arrays** (device kernel): each group points to its aligned activation row
   ```
   BuildStridedPointerArray<<<1, top_k>>>(
       cutlass_a_ptrs, aligned_act_packed,
       aligned_packed_row_stride, top_k);
   ```

10. **CUTLASS down_proj** → `cutlass_down_plan->Run(...)`

11. **Scale + weighted merge** → `ScaleWeightedAccumulateRowsFp32(...)`

### Small Device Kernels Needed

Two trivial kernels:
- `FillPointerArray(void** dst, void* value, int count)` — fill all entries with the same pointer
- `BuildStridedPointerArray(void** dst, void* base, size_t stride, int count)` — dst[i] = base + i*stride

These are ~5 lines each. They replace the host-side loops + H→D uploads.

### What Changes vs Current Code

- Remove all `std::vector<const void*>` host pointer arrays from the dispatch path
- Remove all `DeviceBuffer::CopyFromHost` for pointer arrays in the dispatch path  
- Remove all `cudaMemcpy` D→H for tensor scales
- Pre-allocate `cutlass_a/c/d_ptrs` device buffers in `ExpertLayerSlice::Create()`
- Add `FillPointerArray` and `BuildStridedPointerArray` device kernels to `expert_ops.cu`
- Modify `ScaleRelu2PackRowsToNvfp4` to support aligned row stride (for down_proj activation)
