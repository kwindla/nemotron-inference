#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nemotron/manifest.h"

namespace nemotron {

enum class ArtifactLoadMode {
  kMmap,
  kReadAll,
};

const char* ToString(ArtifactLoadMode mode);

struct ByteRangeView {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;

  bool valid() const;
};

struct AuxiliaryArtifactView {
  std::string name;
  ByteRangeView bytes;
};

struct TensorArtifactView {
  const TensorManifestEntry* manifest_entry = nullptr;
  ByteRangeView packed_bytes;
  std::vector<AuxiliaryArtifactView> auxiliaries;
};

class ArtifactLoader {
 public:
  static std::unique_ptr<ArtifactLoader> OpenVerified(
      const PackedModelManifest& manifest,
      const std::filesystem::path& manifest_path);
  static std::unique_ptr<ArtifactLoader> OpenVerifiedWithMode(
      const PackedModelManifest& manifest,
      const std::filesystem::path& manifest_path,
      ArtifactLoadMode mode);

  ArtifactLoader(ArtifactLoader&&) noexcept;
  ArtifactLoader& operator=(ArtifactLoader&&) noexcept;
  ~ArtifactLoader();

  ArtifactLoader(const ArtifactLoader&) = delete;
  ArtifactLoader& operator=(const ArtifactLoader&) = delete;

  std::optional<TensorArtifactView> FindTensor(const std::string& tensor_name) const;
  const std::filesystem::path& manifest_path() const;
  ArtifactLoadMode load_mode() const;
  std::size_t loaded_file_count() const;
  std::size_t total_loaded_bytes() const;
  std::size_t mapped_file_count() const;
  std::size_t total_mapped_bytes() const;

 private:
  struct Impl;

  explicit ArtifactLoader(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
