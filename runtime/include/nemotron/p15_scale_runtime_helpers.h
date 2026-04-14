// Minimal shared P15 scale helpers used by both the standalone test and the
// live runtime kernel. This header intentionally avoids the broader local-CUTE
// copy/probe machinery so it remains safe to include in the runtime TU.
#pragma once

#include <cstdint>

#include <cute/tensor.hpp>

namespace nemotron::p15_scale_runtime {

using ElementSFCompute = cute::float_ue4m3_t;
constexpr int kScaleGranularityM = 128;
constexpr int kScaleGranularityN = 128;

using AtomScaleLayout = cute::Layout<
    cute::Shape<cute::Shape<cute::_16, cute::_4>>,
    cute::Stride<cute::Stride<cute::_0, cute::_1>>>;

static_assert(cute::size(AtomScaleLayout{}) == 64);
static_assert(cute::cosize(AtomScaleLayout{}) == 4);

struct AtomBaseRows {
  int a_base_row;
  int b_base_row;
};

struct AtomScaleWords {
  std::uint32_t a_word;
  std::uint32_t b_word;
};

CUTE_HOST_DEVICE std::uint8_t encode_scale_byte(float value) {
  return static_cast<std::uint8_t>(ElementSFCompute(value).raw());
}

CUTE_HOST_DEVICE float decode_scale_byte(std::uint8_t raw) {
  return static_cast<float>(ElementSFCompute::bitcast(raw));
}

CUTE_HOST_DEVICE constexpr std::uint32_t pack_scale_word4_bytes(
    std::uint8_t s0,
    std::uint8_t s1,
    std::uint8_t s2,
    std::uint8_t s3) {
  return static_cast<std::uint32_t>(s0) |
         (static_cast<std::uint32_t>(s1) << 8) |
         (static_cast<std::uint32_t>(s2) << 16) |
         (static_cast<std::uint32_t>(s3) << 24);
}

CUTE_HOST_DEVICE constexpr std::uint8_t load_scale_byte(
    std::uint32_t packed_scale_word,
    int byte_index) {
  return static_cast<std::uint8_t>((packed_scale_word >> (byte_index * 8)) & 0xFFu);
}

CUTE_HOST_DEVICE constexpr ElementSFCompute make_scale_element(std::uint8_t raw) {
  return ElementSFCompute::bitcast(static_cast<typename ElementSFCompute::Storage>(raw));
}

template <class AtomCoordTensor>
CUTE_HOST_DEVICE constexpr AtomBaseRows get_atom_base_rows(
    AtomCoordTensor const& c_atom_coords) {
  auto coord0 = c_atom_coords(0);
  return {
      static_cast<int>(cute::get<0>(coord0)),
      static_cast<int>(cute::get<1>(coord0))};
}

CUTE_HOST_DEVICE constexpr bool atom_base_rows_in_bounds(
    AtomBaseRows rows,
    int a_rows,
    int b_rows) {
  return rows.a_base_row >= 0 && rows.a_base_row < a_rows &&
         rows.b_base_row >= 0 && rows.b_base_row < b_rows;
}

template <class AtomCoordTensor>
CUTE_HOST_DEVICE constexpr bool atom_stays_in_single_scale_band(
    AtomCoordTensor const& c_atom_coords,
    int scale_granularity_m = kScaleGranularityM,
    int scale_granularity_n = kScaleGranularityN) {
  int min_row = 1 << 30;
  int max_row = -(1 << 30);
  int min_col = 1 << 30;
  int max_col = -(1 << 30);
  for (int i = 0; i < static_cast<int>(cute::size(c_atom_coords)); ++i) {
    auto coord = c_atom_coords(i);
    const int row = static_cast<int>(cute::get<0>(coord));
    const int col = static_cast<int>(cute::get<1>(coord));
    min_row = row < min_row ? row : min_row;
    max_row = row > max_row ? row : max_row;
    min_col = col < min_col ? col : min_col;
    max_col = col > max_col ? col : max_col;
  }
  return (min_row / scale_granularity_m) == (max_row / scale_granularity_m) &&
         (min_col / scale_granularity_n) == (max_col / scale_granularity_n);
}

template <class AtomCoordTensor>
CUTE_HOST_DEVICE constexpr AtomScaleWords load_atom_scale_words(
    AtomCoordTensor const& c_atom_coords,
    std::uint32_t const* a_scale_words,
    std::uint32_t const* b_scale_words) {
  auto rows = get_atom_base_rows(c_atom_coords);
  return {a_scale_words[rows.a_base_row], b_scale_words[rows.b_base_row]};
}

template <class WordRef>
CUTE_HOST_DEVICE constexpr auto make_atom_scale_tensor(WordRef& packed_word) {
  return cute::make_tensor(
      reinterpret_cast<ElementSFCompute*>(&packed_word),
      AtomScaleLayout{});
}

template <class ScaleTensor>
CUTE_HOST_DEVICE void store_scale_word_k64(
    ScaleTensor const& scale_tensor,
    std::uint32_t packed_scale_word,
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
  constexpr int kScaleBytesPerWord = 4;
  const int logical_cols = static_cast<int>(cute::size<1>(scale_tensor));
  const int segment_len = logical_cols / kScaleBytesPerWord;
  for (int scale_col = 0; scale_col < logical_cols; ++scale_col) {
    const int byte_index = segment_len > 0
        ? min(scale_col / segment_len, kScaleBytesPerWord - 1)
        : 0;
    const std::uint8_t raw = load_scale_byte(packed_scale_word, byte_index);
    scale_tensor(row, scale_col, cute::Int<0>{}) = make_scale_element(raw);
  }
}

}  // namespace nemotron::p15_scale_runtime
