#include <cxxabi.h>
#include <iostream>
#include <memory>
#include <string>

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace {

using ElementAB = cute::float_e2m1_t;
using ElementSFCompute = cute::float_ue4m3_t;
static constexpr int kTmaThreadsRef = 128;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using ElementD = __nv_bfloat16;

std::string Demangle(const char* name) {
  int status = 0;
  std::unique_ptr<char, decltype(&std::free)> demangled(
      abi::__cxa_demangle(name, nullptr, nullptr, &status), &std::free);
  if (status == 0 && demangled) {
    return demangled.get();
  }
  return name;
}

template <class MmaTileShape, class ClusterShape>
struct CollectiveAliases {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      ElementD,
      LayoutC*,
      32,
      ElementD,
      LayoutD*,
      32,
      cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

  using StageCountAutoCarveout =
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassBlockScaledTensorOp,
      cutlass::nv_float4_t<ElementAB>,
      LayoutA*,
      32,
      cutlass::nv_float4_t<ElementAB>,
      LayoutB*,
      32,
      float,
      MmaTileShape,
      ClusterShape,
      StageCountAutoCarveout,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
};

template <class Traits>
void DumpProfile(const char* profile_name) {
  using Mainloop = typename Traits::Mainloop;
  using TiledMma = typename Mainloop::TiledMma;
  using SmemLayoutA = typename Mainloop::SmemLayoutA;
  using SmemLayoutB = typename Mainloop::SmemLayoutB;
  using SmemLayoutSFA = typename Mainloop::SmemLayoutSFA;
  using SmemLayoutSFB = typename Mainloop::SmemLayoutSFB;
  using SmemCopyAtomA = typename Mainloop::SmemCopyAtomA;
  using SmemCopyAtomB = typename Mainloop::SmemCopyAtomB;
  using SmemCopyAtomSFA = typename Mainloop::SmemCopyAtomSFA;
  using SmemCopyAtomSFB = typename Mainloop::SmemCopyAtomSFB;
  using SmemAllocA = typename TiledMma::ValTypeA;
  using SmemAllocB = typename TiledMma::ValTypeB;

  constexpr int math_threads = cute::size(typename TiledMma::ThrLayoutVMNK{});
  constexpr int math_warps = math_threads / 32;
  constexpr int total_threads_ref = kTmaThreadsRef + math_threads;

  constexpr int ab_stages = cute::size<2>(SmemLayoutA{});
  constexpr int sf_stages = cute::size<2>(SmemLayoutSFA{});

  constexpr std::size_t a_stage_elems = cute::cosize(SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{}));
  constexpr std::size_t b_stage_elems = cute::cosize(SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{}));
  constexpr std::size_t sfa_stage_slots = cute::cosize(SmemLayoutSFA{}(cute::_, cute::_, cute::Int<0>{}));
  constexpr std::size_t sfb_stage_slots = cute::cosize(SmemLayoutSFB{}(cute::_, cute::_, cute::Int<0>{}));

  constexpr std::size_t a_stage_bytes = a_stage_elems * sizeof(SmemAllocA);
  constexpr std::size_t b_stage_bytes = b_stage_elems * sizeof(SmemAllocB);
  constexpr std::size_t a_total_bytes = cute::cosize_v<SmemLayoutA> * sizeof(SmemAllocA);
  constexpr std::size_t b_total_bytes = cute::cosize_v<SmemLayoutB> * sizeof(SmemAllocB);

  // The active unified runtime stores execution scales as byte-packed swizzled tensors.
  constexpr std::size_t sfa_stage_bytes_runtime = sfa_stage_slots;
  constexpr std::size_t sfb_stage_bytes_runtime = sfb_stage_slots;
  constexpr std::size_t sfa_total_bytes_runtime = cute::cosize_v<SmemLayoutSFA>;
  constexpr std::size_t sfb_total_bytes_runtime = cute::cosize_v<SmemLayoutSFB>;

  std::cout << "profile=" << profile_name << "\n";
  std::cout << "  mma_tile_mnk=("
            << cute::tile_size<0>(TiledMma{}) << ","
            << cute::tile_size<1>(TiledMma{}) << ","
            << cute::tile_size<2>(TiledMma{}) << ")\n";
  std::cout << "  math_threads=" << math_threads << "\n";
  std::cout << "  math_warps=" << math_warps << "\n";
  std::cout << "  tma_threads_ref=" << kTmaThreadsRef << "\n";
  std::cout << "  total_threads_per_block_ref=" << total_threads_ref << "\n";
  std::cout << "  ab_stages=" << ab_stages << "\n";
  std::cout << "  sf_stages=" << sf_stages << "\n";
  std::cout << "  smem_copy_atom_a=" << Demangle(typeid(SmemCopyAtomA).name()) << "\n";
  std::cout << "  smem_copy_atom_b=" << Demangle(typeid(SmemCopyAtomB).name()) << "\n";
  std::cout << "  smem_copy_atom_sfa=" << Demangle(typeid(SmemCopyAtomSFA).name()) << "\n";
  std::cout << "  smem_copy_atom_sfb=" << Demangle(typeid(SmemCopyAtomSFB).name()) << "\n";
  std::cout << "  a_stage_bytes=" << a_stage_bytes << "\n";
  std::cout << "  b_stage_bytes=" << b_stage_bytes << "\n";
  std::cout << "  sfa_stage_bytes_runtime=" << sfa_stage_bytes_runtime << "\n";
  std::cout << "  sfb_stage_bytes_runtime=" << sfb_stage_bytes_runtime << "\n";
  std::cout << "  single_stage_runtime_bytes="
            << (a_stage_bytes + b_stage_bytes + sfa_stage_bytes_runtime + sfb_stage_bytes_runtime) << "\n";
  std::cout << "  a_total_bytes=" << a_total_bytes << "\n";
  std::cout << "  b_total_bytes=" << b_total_bytes << "\n";
  std::cout << "  sfa_total_bytes_runtime=" << sfa_total_bytes_runtime << "\n";
  std::cout << "  sfb_total_bytes_runtime=" << sfb_total_bytes_runtime << "\n";
  std::cout << "  multistage_runtime_bytes="
            << (a_total_bytes + b_total_bytes + sfa_total_bytes_runtime + sfb_total_bytes_runtime) << "\n";
}

}  // namespace

int main() {
  using P5MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
  using P13MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>;
  using P15MmaTileShape = cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>;
  using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;

  using P5Traits = CollectiveAliases<P5MmaTileShape, ClusterShape>;
  using P13Traits = CollectiveAliases<P13MmaTileShape, ClusterShape>;
  using P15Traits = CollectiveAliases<P15MmaTileShape, ClusterShape>;

  std::cout << "reference_barrier_full="
            << Demangle(typeid(cutlass::arch::ClusterTransactionBarrier).name()) << "\n";
  std::cout << "reference_barrier_empty="
            << Demangle(typeid(cutlass::arch::ClusterBarrier).name()) << "\n";
  std::cout << "reference_named_barrier_math_threads=256\n";
  std::cout << "reference_store_barrier_slots=1\n";

  DumpProfile<P5Traits>("P5");
  // P12 uses the same 128x128x128 builder tile as P5, so its transport
  // contract is expected to match P5 exactly in this dump.
  DumpProfile<P5Traits>("P12");
  DumpProfile<P13Traits>("P13");
  DumpProfile<P15Traits>("P15");
  return 0;
}
