#include "routed_p5_tma_descriptor.cuh"

#include <vector>

namespace nemotron::routed_p5_tma {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

}  // namespace

void* CreateDeviceP5TmaLoadAArray(
    const std::uint8_t* packed_data,
    std::size_t expert_count,
    std::size_t expert_packed_stride_bytes,
    std::size_t output_rows,
    std::size_t input_cols) {
  if (packed_data == nullptr ||
      expert_count == 0 ||
      expert_packed_stride_bytes == 0 ||
      output_rows == 0 ||
      input_cols == 0 ||
      (output_rows % 128u) != 0 ||
      (input_cols % 128u) != 0 ||
      (expert_packed_stride_bytes % sizeof(std::uint8_t)) != 0) {
    return nullptr;
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0) {
    return nullptr;
  }

  std::vector<P5TmaLoadA> host_descriptors;
  host_descriptors.reserve(expert_count);
  for (std::size_t expert_index = 0; expert_index < expert_count; ++expert_index) {
    const std::uint8_t* expert_packed =
        packed_data + (expert_index * expert_packed_stride_bytes);
    auto tensor_a = cute::make_tensor(
        cute::make_gmem_ptr(cute::recast_ptr<ElementAB>(expert_packed)),
        cute::make_layout(
            cute::make_shape(
                static_cast<int32_t>(output_rows),
                static_cast<int32_t>(input_cols),
                int32_t{1}),
            cute::make_stride(
                static_cast<int64_t>(input_cols),
                cute::Int<1>{},
                static_cast<int64_t>(output_rows * input_cols))));
    host_descriptors.push_back(MakeP5TmaLoadA(tensor_a));
  }

  P5TmaLoadA* device_descriptors = nullptr;
  const std::size_t descriptor_bytes = sizeof(P5TmaLoadA) * host_descriptors.size();
  if (!CheckCuda(cudaMalloc(
          reinterpret_cast<void**>(&device_descriptors),
          descriptor_bytes))) {
    return nullptr;
  }
  if (!CheckCuda(cudaMemcpy(
          device_descriptors,
          host_descriptors.data(),
          descriptor_bytes,
          cudaMemcpyHostToDevice))) {
    cudaFree(device_descriptors);
    return nullptr;
  }

  return device_descriptors;
}

void* CreateDeviceP5TmaLoadSFAArray(
    const std::uint8_t* scale_data,
    std::size_t expert_count,
    std::size_t expert_scale_stride_bytes,
    std::size_t output_rows,
    std::size_t input_cols) {
  if (scale_data == nullptr ||
      expert_count == 0 ||
      expert_scale_stride_bytes == 0 ||
      output_rows == 0 ||
      input_cols == 0 ||
      (output_rows % 128u) != 0 ||
      (input_cols % 128u) != 0) {
    return nullptr;
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0) {
    return nullptr;
  }

  std::vector<P5TmaLoadSFA> host_descriptors;
  host_descriptors.reserve(expert_count);
  for (std::size_t expert_index = 0; expert_index < expert_count; ++expert_index) {
    const std::uint8_t* expert_scales =
        scale_data + (expert_index * expert_scale_stride_bytes);
    auto tensor_sfa = cute::make_tensor(
        cute::make_gmem_ptr(
            reinterpret_cast<ElementSF const*>(expert_scales)),
        MakeP5ScaleLayoutSFA(
            static_cast<int32_t>(output_rows),
            static_cast<int32_t>(input_cols)));
    host_descriptors.push_back(MakeP5TmaLoadSFA(tensor_sfa));
  }

  P5TmaLoadSFA* device_descriptors = nullptr;
  const std::size_t descriptor_bytes = sizeof(P5TmaLoadSFA) * host_descriptors.size();
  if (!CheckCuda(cudaMalloc(
          reinterpret_cast<void**>(&device_descriptors),
          descriptor_bytes))) {
    return nullptr;
  }
  if (!CheckCuda(cudaMemcpy(
          device_descriptors,
          host_descriptors.data(),
          descriptor_bytes,
          cudaMemcpyHostToDevice))) {
    cudaFree(device_descriptors);
    return nullptr;
  }

  return device_descriptors;
}

void DestroyDeviceP5TmaLoadAArray(void** descriptor_array) {
  if (descriptor_array == nullptr || *descriptor_array == nullptr) {
    return;
  }
  cudaFree(*descriptor_array);
  *descriptor_array = nullptr;
}

void DestroyDeviceP5TmaLoadSFAArray(void** descriptor_array) {
  DestroyDeviceP5TmaLoadAArray(descriptor_array);
}

std::size_t P5TmaLoadABytes() {
  return sizeof(P5TmaLoadA);
}

std::size_t P5TmaLoadSFABytes() {
  return sizeof(P5TmaLoadSFA);
}

}  // namespace nemotron::routed_p5_tma
