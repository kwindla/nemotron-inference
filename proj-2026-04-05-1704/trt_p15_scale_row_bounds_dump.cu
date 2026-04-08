#include <iostream>
#include <limits>

#include <cuda_bf16.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

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
constexpr int kScaleGranularityM = 128;
constexpr int kScaleGranularityN = 128;

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

struct Stats {
  bool a_base_bounds_ok = true;
  bool b_base_bounds_ok = true;
  bool a_band_uniform = true;
  bool b_band_uniform = true;
  int min_a_row = std::numeric_limits<int>::max();
  int max_a_row = std::numeric_limits<int>::min();
  int min_b_row = std::numeric_limits<int>::max();
  int max_b_row = std::numeric_limits<int>::min();
  int first_bad_thread = -1;
  int first_bad_m = -1;
  int first_bad_n = -1;
};

}  // namespace

int main() {
  using namespace cute;

  TiledMma tiled_mma;
  auto dense_c = make_identity_tensor(make_shape(Int<256>{}, Int<128>{}));

  Stats stats;
  int sample_printed = 0;

  for (int thread_idx = 0; thread_idx < static_cast<int>(size(tiled_mma)); ++thread_idx) {
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    auto part_c = thread_mma.partition_C(dense_c);
    const int m_tiles = static_cast<int>(size<1>(part_c));
    const int n_tiles = static_cast<int>(size<2>(part_c));

    for (int m = 0; m < m_tiles; ++m) {
      for (int n = 0; n < n_tiles; ++n) {
        auto c_atom = part_c(_, m, n);
        int min_row = std::numeric_limits<int>::max();
        int max_row = std::numeric_limits<int>::min();
        int min_col = std::numeric_limits<int>::max();
        int max_col = std::numeric_limits<int>::min();

        for (int i = 0; i < static_cast<int>(size(c_atom)); ++i) {
          auto coord = c_atom(i);
          const int row = static_cast<int>(get<0>(coord));
          const int col = static_cast<int>(get<1>(coord));
          min_row = min(min_row, row);
          max_row = max(max_row, row);
          min_col = min(min_col, col);
          max_col = max(max_col, col);
        }

        const int a_base_row = static_cast<int>(get<0>(c_atom(0)));
        const int b_base_row = static_cast<int>(get<1>(c_atom(0)));
        const bool a_in_bounds = a_base_row >= 0 && a_base_row < 256;
        const bool b_in_bounds = b_base_row >= 0 && b_base_row < 128;
        const bool a_uniform = (min_row / kScaleGranularityM) == (max_row / kScaleGranularityM);
        const bool b_uniform = (min_col / kScaleGranularityN) == (max_col / kScaleGranularityN);

        stats.a_base_bounds_ok = stats.a_base_bounds_ok && a_in_bounds;
        stats.b_base_bounds_ok = stats.b_base_bounds_ok && b_in_bounds;
        stats.a_band_uniform = stats.a_band_uniform && a_uniform;
        stats.b_band_uniform = stats.b_band_uniform && b_uniform;
        stats.min_a_row = min(stats.min_a_row, min_row);
        stats.max_a_row = max(stats.max_a_row, max_row);
        stats.min_b_row = min(stats.min_b_row, min_col);
        stats.max_b_row = max(stats.max_b_row, max_col);

        if ((!a_in_bounds || !b_in_bounds || !a_uniform || !b_uniform) &&
            stats.first_bad_thread < 0) {
          stats.first_bad_thread = thread_idx;
          stats.first_bad_m = m;
          stats.first_bad_n = n;
        }

        if (thread_idx == 0 && sample_printed < 8) {
          std::cout
              << "sample thread=0 m=" << m
              << " n=" << n
              << " a_base_row=" << a_base_row
              << " b_base_row=" << b_base_row
              << " min_row=" << min_row
              << " max_row=" << max_row
              << " min_col=" << min_col
              << " max_col=" << max_col
              << " a_band_uniform=" << a_uniform
              << " b_band_uniform=" << b_uniform
              << "\n";
          ++sample_printed;
        }
      }
    }
  }

  std::cout << "a_base_bounds_ok=" << stats.a_base_bounds_ok << "\n";
  std::cout << "b_base_bounds_ok=" << stats.b_base_bounds_ok << "\n";
  std::cout << "a_band_uniform=" << stats.a_band_uniform << "\n";
  std::cout << "b_band_uniform=" << stats.b_band_uniform << "\n";
  std::cout << "min_a_row=" << stats.min_a_row << "\n";
  std::cout << "max_a_row=" << stats.max_a_row << "\n";
  std::cout << "min_b_row=" << stats.min_b_row << "\n";
  std::cout << "max_b_row=" << stats.max_b_row << "\n";
  std::cout << "first_bad_thread=" << stats.first_bad_thread << "\n";
  std::cout << "first_bad_m=" << stats.first_bad_m << "\n";
  std::cout << "first_bad_n=" << stats.first_bad_n << "\n";

  return 0;
}
