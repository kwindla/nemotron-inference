#include <iostream>
#include <memory>
#include <type_traits>
#include <typeinfo>

#include <cuda_bf16.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

using ArchTag = cutlass::arch::Sm120;
using OutputType = __nv_bfloat16;
constexpr bool IsSM120 = true;
using InputClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using ClusterShape = InputClusterShape;
using MmaTileShape = cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>;
using ElementAct = cutlass::float_e2m1_t;
using ElementWeight = cutlass::float_e2m1_t;
using ElementD = OutputType;
using ElementCSafe = ElementD;
using ElementAccumulator = float;
using ElementActBlockScaled = cutlass::nv_float4_t<ElementAct>;
using ElementWeightBlockScaled = cutlass::nv_float4_t<ElementWeight>;
constexpr int AlignmentAct = 128 / cutlass::sizeof_bits<ElementAct>::value;
constexpr int AlignmentWeight = 128 / cutlass::sizeof_bits<ElementWeight>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementAct>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using EpilogueScheduleSM90 = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using EpilogueScheduleSM120 = cutlass::epilogue::TmaWarpSpecialized;
using EpilogueSchedule = std::conditional_t<IsSM120, EpilogueScheduleSM120, EpilogueScheduleSM90>;
using EpilogueTensorOp = cutlass::arch::OpClassBlockScaledTensorOp;
using EpilogueSubTile = cutlass::epilogue::collective::EpilogueTileAuto;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    EpilogueTensorOp,
    MmaTileShape,
    ClusterShape,
    EpilogueSubTile,
    ElementAccumulator,
    ElementAccumulator,
    ElementCSafe,
    LayoutC*,
    AlignmentC,
    ElementD,
    LayoutD*,
    AlignmentD,
    EpilogueSchedule>::CollectiveOp;
using StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
using TensorOp = cutlass::arch::OpClassBlockScaledTensorOp;
using MainloopElementAct = ElementActBlockScaled;
using MainloopElementWeight = ElementWeightBlockScaled;
using SwappedMainloopElementA = MainloopElementWeight;
using SwappedMainloopElementB = MainloopElementAct;
constexpr int SwappedAlignmentA = AlignmentWeight;
constexpr int SwappedAlignmentB = AlignmentAct;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    TensorOp,
    SwappedMainloopElementA,
    LayoutA*,
    SwappedAlignmentA,
    SwappedMainloopElementB,
    LayoutB*,
    SwappedAlignmentB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    StageCountAutoCarveout,
    KernelSchedule>::CollectiveOp;
using TiledMma = typename CollectiveMainloop::TiledMma;
using SmemCopyAtomA = typename CollectiveMainloop::SmemCopyAtomA;
using SmemCopyAtomB = typename CollectiveMainloop::SmemCopyAtomB;
using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;

template <class T>
std::string TypeName() {
#if defined(__GNUG__)
  int status = 0;
  std::unique_ptr<char, decltype(&std::free)> demangled(
      abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status), &std::free);
  if (status == 0 && demangled) {
    return demangled.get();
  }
#endif
  return typeid(T).name();
}

template <class Tensor>
void DumpTensor(const char* name, Tensor const&) {
  using T = std::remove_cvref_t<Tensor>;
  std::cout << "TENSOR " << name << "\n";
  std::cout << "  engine=" << TypeName<typename T::engine_type>() << "\n";
  std::cout << "  iterator=" << TypeName<typename T::iterator>() << "\n";
  std::cout << "  value_type=" << TypeName<typename T::value_type>() << "\n";
  std::cout << "  element_type=" << TypeName<typename T::element_type>() << "\n";
  std::cout << "  is_rmem=" << cute::is_rmem<typename T::iterator>::value << "\n";
  std::cout << "  is_smem=" << cute::is_smem<typename T::iterator>::value << "\n";
  std::cout << "  rank=" << T::rank << "\n";
  std::cout << "  size0=" << cute::size<0>(typename T::layout_type{}) << "\n";
  std::cout << "  size1=" << cute::size<1>(typename T::layout_type{}) << "\n";
  std::cout << "  size2=" << cute::size<2>(typename T::layout_type{}) << "\n";
}

int main() {
  using namespace cute;

  auto sA = make_tensor(make_smem_ptr(static_cast<ElementAct*>(nullptr)), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(static_cast<ElementWeight*>(nullptr)), SmemLayoutB{});
  auto tiled_mma = TiledMma{};
  auto thread_mma = tiled_mma.get_thread_slice(0);

  auto tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
  auto tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));

  auto smem_tiled_copy_a = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(0);
  auto tCrA_copy_view = smem_thr_copy_a.retile_D(tCrA);

  auto smem_tiled_copy_b = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
  auto smem_thr_copy_b = smem_tiled_copy_b.get_thread_slice(0);
  auto tCrB_copy_view = smem_thr_copy_b.retile_D(tCrB);

  DumpTensor("tCrA", tCrA);
  DumpTensor("tCrB", tCrB);
  DumpTensor("tCrA_copy_view", tCrA_copy_view);
  DumpTensor("tCrB_copy_view", tCrB_copy_view);

  return 0;
}
