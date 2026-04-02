#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define NEMOTRON_FLASHINFER_MOE_PLUGIN_ABI_VERSION 2

enum {
  NEMOTRON_FLASHINFER_WEIGHT_LAYOUT_RAW_ROW_MAJOR = 0,
  NEMOTRON_FLASHINFER_WEIGHT_LAYOUT_SHUFFLED_MAJOR_K = 1,
};

enum {
  NEMOTRON_FLASHINFER_SCALE_LAYOUT_RAW_LINEAR = 0,
  NEMOTRON_FLASHINFER_SCALE_LAYOUT_SWIZZLED_128X4 = 1,
};

enum {
  NEMOTRON_FLASHINFER_BACKEND_KIND_UNKNOWN = 0,
  NEMOTRON_FLASHINFER_BACKEND_KIND_TRT_SPLIT = 1,
  NEMOTRON_FLASHINFER_BACKEND_KIND_CUTLASS_FUSED = 2,
};

typedef struct NemotronFlashInferNvfp4WeightView {
  const std::uint8_t* packed_data;
  std::size_t packed_nbytes;
  const std::uint8_t* scale_data;
  std::size_t scale_nbytes;
  float tensor_scale;
  float dequant_scale;
  std::size_t output_rows;
  std::size_t input_cols;
  std::size_t scale_rows;
  std::size_t scale_cols;
  std::int32_t packed_layout;
  std::int32_t scale_layout;
} NemotronFlashInferNvfp4WeightView;

typedef struct NemotronFlashInferRoutedMoECreateParams {
  std::size_t layer_index;
  std::size_t num_experts;
  std::size_t hidden_size;
  std::size_t moe_latent_size;
  std::size_t intermediate_size;
  std::size_t top_k;
  std::size_t n_group;
  std::size_t topk_group;
  float routed_scaling_factor;
  std::int32_t norm_topk_prob;
  std::int64_t routing_method_type;
  std::int64_t activation_type;
  const NemotronFlashInferNvfp4WeightView* up_experts;
  const NemotronFlashInferNvfp4WeightView* down_experts;
} NemotronFlashInferRoutedMoECreateParams;

typedef struct NemotronFlashInferRoutedMoERunParams {
  const float* hidden_states_fp32;
  std::size_t token_count;
  const std::int32_t* topk_ids_device;
  const float* topk_weights_device;
  float* output_fp32;
  void* cuda_stream;
} NemotronFlashInferRoutedMoERunParams;

typedef struct NemotronFlashInferRoutedMoEHandle NemotronFlashInferRoutedMoEHandle;

typedef int (*NemotronFlashInferAbiVersionFn)();
typedef int (*NemotronFlashInferBackendKindFn)();
typedef NemotronFlashInferRoutedMoEHandle* (*NemotronFlashInferRoutedMoECreateFn)(
    const NemotronFlashInferRoutedMoECreateParams*);
typedef void (*NemotronFlashInferRoutedMoEDestroyFn)(NemotronFlashInferRoutedMoEHandle*);
typedef bool (*NemotronFlashInferRoutedMoERunFn)(
    NemotronFlashInferRoutedMoEHandle*,
    const NemotronFlashInferRoutedMoERunParams*);
typedef const char* (*NemotronFlashInferLastErrorFn)();

#ifdef __cplusplus
}
#endif
