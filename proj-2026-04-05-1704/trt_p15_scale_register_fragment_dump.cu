#include <iostream>

#include <cuda_bf16.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

#include "nemotron/p15_scale_helpers.h"

namespace {

namespace cute = ::cute;

using ArchTag = cutlass::arch::Sm120;
using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using MmaTileShape = cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>;
using ElementAB = cutlass::float_e2m1_t;
using ElementD = __nv_bfloat16;
using ElementAccumulator = float;
using ElementABBlockScaled = cutlass::nv_float4_t<ElementAB>;

constexpr int kAlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassBlockScaledTensorOp,
    MmaTileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementD,
    LayoutC*,
    kAlignmentAB,
    ElementD,
    LayoutD*,
    kAlignmentD,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

using StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementABBlockScaled,
    LayoutA*,
    kAlignmentAB,
    ElementABBlockScaled,
    LayoutB*,
    kAlignmentAB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    StageCountAutoCarveout,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using TiledMma = typename CollectiveMainloop::TiledMma;
using SmemLayoutSFA = typename CollectiveMainloop::SmemLayoutSFA;
using SmemLayoutSFB = typename CollectiveMainloop::SmemLayoutSFB;

template <class Tensor>
void DumpTensorSummary(char const* name, Tensor const& tensor) {
  std::cout << name << ".size=" << cute::size(tensor) << "\n";
  std::cout << name << ".cosize=" << cute::cosize(tensor.layout()) << "\n";
  std::cout << name << ".layout=";
  cute::print(tensor.layout());
  std::cout << "\n";
}

}  // namespace

int main() {
  using namespace cute;

  std::cout << "Stages=" << CollectiveMainloop::DispatchPolicy::Stages << "\n";

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(0);

  auto sfa = make_tensor(static_cast<std::uint8_t*>(nullptr), SmemLayoutSFA{});
  auto sfb = make_tensor(static_cast<std::uint8_t*>(nullptr), SmemLayoutSFB{});
  auto sfa_stage0 = sfa(_, _, Int<0>{});
  auto sfb_stage0 = sfb(_, _, Int<0>{});

  auto tCrSFA_raw = nemotron::p15_scale::partition_fragment_SFA(sfa_stage0, tiled_mma, thread_mma);
  auto tCrSFB_raw = nemotron::p15_scale::partition_fragment_SFB(sfb_stage0, tiled_mma, thread_mma);
  DumpTensorSummary("tCrSFA_raw", tCrSFA_raw);
  DumpTensorSummary("tCrSFB_raw", tCrSFB_raw);
  auto tCrSFA_k0 = tCrSFA_raw(_, _, Int<0>{});
  auto tCrSFB_k0 = tCrSFB_raw(_, _, Int<0>{});

  DumpTensorSummary("tCrSFA_raw_k0", tCrSFA_k0);
  DumpTensorSummary("tCrSFB_raw_k0", tCrSFB_k0);

  std::cout << "tCrSFA_raw.rank=" << decltype(tCrSFA_raw)::rank << "\n";
  std::cout << "tCrSFB_raw.rank=" << decltype(tCrSFB_raw)::rank << "\n";
  std::cout << "tCrSFA_raw.shape=("
            << size<0>(tCrSFA_raw) << ","
            << size<1>(tCrSFA_raw) << ","
            << size<2>(tCrSFA_raw) << ")\n";
  std::cout << "tCrSFB_raw.shape=("
            << size<0>(tCrSFB_raw) << ","
            << size<1>(tCrSFB_raw) << ","
            << size<2>(tCrSFB_raw) << ")\n";

  return 0;
}
