#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
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
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;

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

template <class TensorA>
auto MakeP5TmaLoadA(TensorA const& tensor_a) {
  return cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_a,
      SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<128>{}, cute::Int<128>{}),
      cute::_1{});
}

using P5TmaLoadA = decltype(MakeP5TmaLoadA(cute::make_tensor(
    cute::make_gmem_ptr(static_cast<ElementAB const*>(nullptr)),
    cute::make_layout(
        cute::make_shape(int32_t{128}, int32_t{128}, int32_t{1}),
        cute::make_stride(int64_t{128}, cute::Int<1>{}, int64_t{16384})))));

void* CreateDeviceP5TmaLoadAArray(
    const std::uint8_t* packed_data,
    std::size_t expert_count,
    std::size_t expert_packed_stride_bytes,
    std::size_t output_rows,
    std::size_t input_cols);

void DestroyDeviceP5TmaLoadAArray(void** descriptor_array);

}  // namespace nemotron::routed_p5_tma
