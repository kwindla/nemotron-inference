#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"

#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"

namespace nemotron::routed_p5_tma {

using ElementAB = cutlass::float_e2m1_t;
using ElementAccumulator = float;
using ElementD = __nv_bfloat16;
using ElementABBlockScaled = cutlass::nv_float4_t<ElementAB>;
using ElementSF = cutlass::float_ue4m3_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

constexpr int kAlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    MmaTileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementD,
    LayoutC*,
    kAlignmentD,
    ElementD,
    LayoutD*,
    kAlignmentD,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

using StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementABBlockScaled, LayoutA*, kAlignmentAB,
    ElementABBlockScaled, LayoutB*, kAlignmentAB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    StageCountAutoCarveout,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
using SmemLayoutSFA = typename CollectiveMainloop::SmemLayoutSFA;
using SmemLayoutSFB = typename CollectiveMainloop::SmemLayoutSFB;

CUTE_HOST_DEVICE constexpr auto MakeP5ScaleLayoutSFA(
    int32_t output_rows,
    int32_t input_cols) {
  return ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(output_rows, int32_t{1}, input_cols, int32_t{1}));
}

CUTE_HOST_DEVICE constexpr auto MakeP5ScaleLayoutSFB(
    int32_t token_rows,
    int32_t input_cols) {
  return ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(int32_t{1}, token_rows, input_cols, int32_t{1}));
}

template <class TensorA>
auto MakeP5TmaLoadA(TensorA const& tensor_a) {
  return cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_a,
      SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<128>{}, cute::Int<128>{}),
      cute::_1{});
}

template <class TensorB>
auto MakeP5TmaLoadB(TensorB const& tensor_b) {
  return cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_b,
      SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<128>{}, cute::Int<128>{}),
      cute::_1{});
}

template <class TensorSFA>
auto MakeP5TmaLoadSFA(TensorSFA const& tensor_sfa) {
  return cute::make_tma_copy<uint16_t>(
      cute::SM90_TMA_LOAD{},
      tensor_sfa,
      SmemLayoutSFA{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<128>{}, cute::Int<128>{}),
      cute::_1{});
}

template <class TensorSFB>
auto MakeP5TmaLoadSFB(TensorSFB const& tensor_sfb) {
  return cute::make_tma_copy<uint16_t>(
      cute::SM90_TMA_LOAD{},
      tensor_sfb,
      SmemLayoutSFB{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<128>{}, cute::Int<128>{}),
      cute::_1{});
}

using P5TmaLoadA = decltype(MakeP5TmaLoadA(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<ElementAB const*>(nullptr)),
    cute::make_layout(
        cute::make_shape(int32_t{128}, int32_t{128}, int32_t{1}),
        cute::make_stride(int64_t{128}, cute::Int<1>{}, int64_t{16384})))));

using P5TmaLoadB = decltype(MakeP5TmaLoadB(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<ElementAB const*>(nullptr)),
    cute::make_layout(
        cute::make_shape(int32_t{128}, int32_t{128}, int32_t{1}),
        cute::make_stride(int64_t{128}, cute::Int<1>{}, int64_t{16384})))));

using P5TmaLoadSFA = decltype(MakeP5TmaLoadSFA(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<ElementSF const*>(nullptr)),
    MakeP5ScaleLayoutSFA(int32_t{128}, int32_t{128}))));

using P5TmaLoadSFB = decltype(MakeP5TmaLoadSFB(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<ElementSF const*>(nullptr)),
    MakeP5ScaleLayoutSFB(int32_t{128}, int32_t{128}))));

void* CreateDeviceP5TmaLoadAArray(
    const std::uint8_t* packed_data,
    std::size_t expert_count,
    std::size_t expert_packed_stride_bytes,
    std::size_t output_rows,
    std::size_t input_cols);

void* CreateDeviceP5TmaLoadSFAArray(
    const std::uint8_t* scale_data,
    std::size_t expert_count,
    std::size_t expert_scale_stride_bytes,
    std::size_t output_rows,
    std::size_t input_cols);

// Builds per-expert TMA descriptors that load the weight as the B operand
// (N-dim = output_rows). Used by the swap=false unified kernel where the
// input is the A operand and the weight is B. Same gmem layout as the
// existing weight-as-A path; only the destination smem layout differs.
void* CreateDeviceP5TmaLoadBArray(
    const std::uint8_t* packed_data,
    std::size_t expert_count,
    std::size_t expert_packed_stride_bytes,
    std::size_t output_rows,
    std::size_t input_cols);

// Per-expert TMA descriptors that load the weight scales as SFB.
void* CreateDeviceP5TmaLoadSFBArray(
    const std::uint8_t* scale_data,
    std::size_t expert_count,
    std::size_t expert_scale_stride_bytes,
    std::size_t output_rows,
    std::size_t input_cols);

void DestroyDeviceP5TmaLoadAArray(void** descriptor_array);
void DestroyDeviceP5TmaLoadSFAArray(void** descriptor_array);
void DestroyDeviceP5TmaLoadBArray(void** descriptor_array);
void DestroyDeviceP5TmaLoadSFBArray(void** descriptor_array);

std::size_t P5TmaLoadABytes();
std::size_t P5TmaLoadSFABytes();
std::size_t P5TmaLoadBBytes();
std::size_t P5TmaLoadSFBBytes();

}  // namespace nemotron::routed_p5_tma
