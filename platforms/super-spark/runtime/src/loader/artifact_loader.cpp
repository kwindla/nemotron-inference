#include "nemotron/artifact_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nemotron {
namespace {

std::filesystem::path ResolveManifestPath(
    const std::filesystem::path& manifest_path,
    const std::string& file) {
  const std::filesystem::path file_path(file);
  if (file_path.is_absolute()) {
    return file_path;
  }
  return manifest_path.parent_path() / file_path;
}

std::string NormalizePathKey(const std::filesystem::path& path) {
  std::error_code ec;
  const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
  if (ec) {
    return path.lexically_normal().string();
  }
  return absolute.lexically_normal().string();
}

bool RangeFits(std::size_t file_size, std::size_t offset_bytes, std::size_t nbytes) {
  return offset_bytes <= file_size && nbytes <= (file_size - offset_bytes);
}

bool ReadAllBytes(int fd, std::size_t file_size, std::vector<std::uint8_t>* bytes) {
  if (fd < 0 || bytes == nullptr || file_size == 0) {
    return false;
  }
  bytes->assign(file_size, 0);
  std::size_t total_read = 0;
  while (total_read < file_size) {
    const ssize_t chunk = read(fd, bytes->data() + total_read, file_size - total_read);
    if (chunk <= 0) {
      bytes->clear();
      return false;
    }
    total_read += static_cast<std::size_t>(chunk);
  }
  return true;
}

}  // namespace

const char* ToString(ArtifactLoadMode mode) {
  switch (mode) {
    case ArtifactLoadMode::kMmap:
      return "mmap";
    case ArtifactLoadMode::kReadAll:
      return "read_all";
  }
  return "unknown";
}

bool ByteRangeView::valid() const {
  return data != nullptr && size != 0;
}

struct ArtifactLoader::Impl {
  struct MappedFile {
    ArtifactLoadMode load_mode = ArtifactLoadMode::kMmap;
    std::filesystem::path path;
    int fd = -1;
    std::size_t size = 0;
    const std::uint8_t* data = nullptr;
    std::vector<std::uint8_t> owned_bytes;

    void Reset() {
      if (load_mode == ArtifactLoadMode::kMmap && data != nullptr && size != 0) {
        munmap(const_cast<std::uint8_t*>(data), size);
      }
      if (fd >= 0) {
        close(fd);
      }
      fd = -1;
      size = 0;
      data = nullptr;
      owned_bytes.clear();
      path.clear();
    }
  };

  explicit Impl(
      const PackedModelManifest& loaded_manifest,
      std::filesystem::path loaded_manifest_path,
      ArtifactLoadMode requested_load_mode,
      bool requested_prefetch_mapped_files)
      : manifest_path(std::move(loaded_manifest_path)),
        manifest(loaded_manifest),
        load_mode(requested_load_mode),
        prefetch_mapped_files(requested_prefetch_mapped_files) {}

  ~Impl() {
    for (MappedFile& mapped_file : mapped_files) {
      mapped_file.Reset();
    }
  }

  std::optional<ByteRangeView> ViewRange(
      const std::filesystem::path& file_path,
      std::size_t offset_bytes,
      std::size_t nbytes) const {
    const std::string key = NormalizePathKey(file_path);
    auto it = mapped_file_indices.find(key);
    if (it == mapped_file_indices.end()) {
      return std::nullopt;
    }

    const MappedFile& mapped_file = mapped_files[it->second];
    if (!RangeFits(mapped_file.size, offset_bytes, nbytes)) {
      return std::nullopt;
    }

    return ByteRangeView{
        mapped_file.data + offset_bytes,
        nbytes,
    };
  }

  bool MapFile(const std::filesystem::path& file_path) {
    const std::string key = NormalizePathKey(file_path);
    if (mapped_file_indices.find(key) != mapped_file_indices.end()) {
      return true;
    }

    std::error_code ec;
    const std::uintmax_t file_size = std::filesystem::file_size(file_path, ec);
    if (ec || file_size == 0 || file_size > static_cast<std::uintmax_t>(SIZE_MAX)) {
      return false;
    }

    const int fd = open(file_path.c_str(), O_RDONLY);
    if (fd < 0) {
      return false;
    }

    MappedFile mapped_file;
    mapped_file.load_mode = load_mode;
    mapped_file.path = file_path;
    mapped_file.fd = fd;
    mapped_file.size = static_cast<std::size_t>(file_size);
    if (load_mode == ArtifactLoadMode::kMmap) {
      void* mapped = mmap(nullptr, mapped_file.size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (mapped == MAP_FAILED) {
        close(fd);
        return false;
      }
#ifdef POSIX_FADV_WILLNEED
      if (prefetch_mapped_files) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
      }
#endif
#ifdef MADV_SEQUENTIAL
      if (prefetch_mapped_files) {
        madvise(mapped, mapped_file.size, MADV_SEQUENTIAL);
      }
#endif
#ifdef MADV_WILLNEED
      if (prefetch_mapped_files) {
        madvise(mapped, mapped_file.size, MADV_WILLNEED);
      }
#endif
      mapped_file.data = static_cast<const std::uint8_t*>(mapped);
    } else {
#ifdef POSIX_FADV_SEQUENTIAL
      posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
      if (!ReadAllBytes(fd, mapped_file.size, &mapped_file.owned_bytes)) {
        close(fd);
        return false;
      }
      mapped_file.data = mapped_file.owned_bytes.data();
    }

    mapped_file_indices.emplace(key, mapped_files.size());
    total_mapped_bytes += mapped_file.size;
    mapped_files.push_back(std::move(mapped_file));
    return true;
  }

  std::filesystem::path manifest_path;
  PackedModelManifest manifest;
  ArtifactLoadMode load_mode = ArtifactLoadMode::kMmap;
  bool prefetch_mapped_files = false;
  std::vector<MappedFile> mapped_files;
  std::unordered_map<std::string, std::size_t> mapped_file_indices;
  std::unordered_map<std::string, std::size_t> tensor_indices;
  std::size_t total_mapped_bytes = 0;
};

std::unique_ptr<ArtifactLoader> ArtifactLoader::OpenVerified(
    const PackedModelManifest& manifest,
    const std::filesystem::path& manifest_path) {
  return OpenVerifiedWithMode(manifest, manifest_path, ArtifactLoadMode::kMmap);
}

std::unique_ptr<ArtifactLoader> ArtifactLoader::OpenVerifiedWithMode(
    const PackedModelManifest& manifest,
    const std::filesystem::path& manifest_path,
    ArtifactLoadMode mode,
    bool prefetch_mapped_files) {
  if (HasManifestErrors(ValidateManifest(manifest))) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>(manifest, manifest_path, mode, prefetch_mapped_files);
  for (std::size_t tensor_index = 0; tensor_index < impl->manifest.tensors.size(); ++tensor_index) {
    const TensorManifestEntry& tensor = impl->manifest.tensors[tensor_index];
    impl->tensor_indices.emplace(tensor.name, tensor_index);

    const std::filesystem::path packed_path = ResolveManifestPath(manifest_path, tensor.packed_file);
    if (!impl->MapFile(packed_path) ||
        !impl->ViewRange(packed_path, tensor.offset_bytes, tensor.nbytes).has_value()) {
      return nullptr;
    }

    for (const TensorAuxiliaryManifest& auxiliary : tensor.auxiliaries) {
      const std::filesystem::path auxiliary_path =
          ResolveManifestPath(manifest_path, auxiliary.file);
      if (!impl->MapFile(auxiliary_path) ||
          !impl->ViewRange(auxiliary_path, auxiliary.offset_bytes, auxiliary.nbytes).has_value()) {
        return nullptr;
      }
    }
  }

  return std::unique_ptr<ArtifactLoader>(new ArtifactLoader(std::move(impl)));
}

ArtifactLoader::ArtifactLoader(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ArtifactLoader::ArtifactLoader(ArtifactLoader&&) noexcept = default;
ArtifactLoader& ArtifactLoader::operator=(ArtifactLoader&&) noexcept = default;
ArtifactLoader::~ArtifactLoader() = default;

std::optional<TensorArtifactView> ArtifactLoader::FindTensor(const std::string& tensor_name) const {
  auto tensor_it = impl_->tensor_indices.find(tensor_name);
  if (tensor_it == impl_->tensor_indices.end()) {
    return std::nullopt;
  }

  const TensorManifestEntry& tensor = impl_->manifest.tensors[tensor_it->second];
  const std::filesystem::path packed_path = ResolveManifestPath(impl_->manifest_path, tensor.packed_file);
  const std::optional<ByteRangeView> packed_view =
      impl_->ViewRange(packed_path, tensor.offset_bytes, tensor.nbytes);
  if (!packed_view.has_value()) {
    return std::nullopt;
  }

  TensorArtifactView view;
  view.manifest_entry = &tensor;
  view.packed_bytes = *packed_view;
  for (const TensorAuxiliaryManifest& auxiliary : tensor.auxiliaries) {
    const std::filesystem::path auxiliary_path =
        ResolveManifestPath(impl_->manifest_path, auxiliary.file);
    const std::optional<ByteRangeView> auxiliary_view =
        impl_->ViewRange(auxiliary_path, auxiliary.offset_bytes, auxiliary.nbytes);
    if (!auxiliary_view.has_value()) {
      return std::nullopt;
    }
    view.auxiliaries.push_back(AuxiliaryArtifactView{
        auxiliary.name,
        *auxiliary_view,
    });
  }

  return view;
}

const std::filesystem::path& ArtifactLoader::manifest_path() const {
  return impl_->manifest_path;
}

ArtifactLoadMode ArtifactLoader::load_mode() const {
  return impl_->load_mode;
}

std::size_t ArtifactLoader::loaded_file_count() const {
  return impl_->mapped_files.size();
}

std::size_t ArtifactLoader::total_loaded_bytes() const {
  return impl_->total_mapped_bytes;
}

std::size_t ArtifactLoader::mapped_file_count() const {
  return loaded_file_count();
}

std::size_t ArtifactLoader::total_mapped_bytes() const {
  return total_loaded_bytes();
}

}  // namespace nemotron
