#include "nemotron/manifest.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>

namespace nemotron {
namespace {

struct JsonValue {
  using Array = std::vector<JsonValue>;
  using Object = std::unordered_map<std::string, JsonValue>;
  using Storage = std::variant<std::nullptr_t, bool, std::int64_t, std::string, Array, Object>;

  Storage storage;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : text_(text) {}

  JsonValue Parse() {
    JsonValue value = ParseValue();
    SkipWhitespace();
    if (!AtEnd()) {
      throw std::runtime_error("unexpected trailing content at byte " + std::to_string(pos_));
    }
    return value;
  }

 private:
  JsonValue ParseValue() {
    SkipWhitespace();
    if (AtEnd()) {
      throw std::runtime_error("unexpected end of input");
    }

    const char ch = text_[pos_];
    if (ch == '{') {
      return ParseObject();
    }
    if (ch == '[') {
      return ParseArray();
    }
    if (ch == '"') {
      return JsonValue{ParseString()};
    }
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
      return JsonValue{ParseInteger()};
    }
    if (ConsumeLiteral("true")) {
      return JsonValue{true};
    }
    if (ConsumeLiteral("false")) {
      return JsonValue{false};
    }
    if (ConsumeLiteral("null")) {
      return JsonValue{nullptr};
    }

    throw std::runtime_error("unexpected token at byte " + std::to_string(pos_));
  }

  JsonValue ParseObject() {
    Expect('{');
    JsonValue::Object object;
    SkipWhitespace();
    if (TryConsume('}')) {
      return JsonValue{std::move(object)};
    }

    while (true) {
      SkipWhitespace();
      if (AtEnd() || text_[pos_] != '"') {
        throw std::runtime_error("expected object key string at byte " + std::to_string(pos_));
      }
      std::string key = ParseString();
      SkipWhitespace();
      Expect(':');
      object.emplace(std::move(key), ParseValue());
      SkipWhitespace();
      if (TryConsume('}')) {
        break;
      }
      Expect(',');
    }

    return JsonValue{std::move(object)};
  }

  JsonValue ParseArray() {
    Expect('[');
    JsonValue::Array array;
    SkipWhitespace();
    if (TryConsume(']')) {
      return JsonValue{std::move(array)};
    }

    while (true) {
      array.push_back(ParseValue());
      SkipWhitespace();
      if (TryConsume(']')) {
        break;
      }
      Expect(',');
    }

    return JsonValue{std::move(array)};
  }

  std::string ParseString() {
    Expect('"');
    std::string result;
    while (!AtEnd()) {
      const char ch = text_[pos_++];
      if (ch == '"') {
        return result;
      }
      if (ch != '\\') {
        result.push_back(ch);
        continue;
      }
      if (AtEnd()) {
        throw std::runtime_error("unterminated escape sequence at byte " + std::to_string(pos_));
      }

      const char escaped = text_[pos_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          result.push_back(escaped);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u':
          throw std::runtime_error("unicode escape sequences are not supported in the current manifest parser");
        default:
          throw std::runtime_error("invalid escape sequence at byte " + std::to_string(pos_ - 1));
      }
    }
    throw std::runtime_error("unterminated string literal");
  }

  std::int64_t ParseInteger() {
    const std::size_t start = pos_;
    if (text_[pos_] == '-') {
      ++pos_;
    }
    if (AtEnd() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      throw std::runtime_error("expected digit at byte " + std::to_string(pos_));
    }
    while (!AtEnd() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    return std::stoll(std::string(text_.substr(start, pos_ - start)));
  }

  void SkipWhitespace() {
    while (!AtEnd() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  void Expect(char expected) {
    SkipWhitespace();
    if (AtEnd() || text_[pos_] != expected) {
      throw std::runtime_error(
          "expected '" + std::string(1, expected) + "' at byte " + std::to_string(pos_));
    }
    ++pos_;
  }

  bool TryConsume(char expected) {
    SkipWhitespace();
    if (!AtEnd() && text_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  bool ConsumeLiteral(std::string_view literal) {
    SkipWhitespace();
    if (text_.substr(pos_, literal.size()) == literal) {
      pos_ += literal.size();
      return true;
    }
    return false;
  }

  bool AtEnd() const {
    return pos_ >= text_.size();
  }

  std::string_view text_;
  std::size_t pos_ = 0;
};

const JsonValue::Object* AsObject(const JsonValue* value) {
  if (value == nullptr) {
    return nullptr;
  }
  return std::get_if<JsonValue::Object>(&value->storage);
}

const JsonValue::Array* AsArray(const JsonValue* value) {
  if (value == nullptr) {
    return nullptr;
  }
  return std::get_if<JsonValue::Array>(&value->storage);
}

const std::string* AsString(const JsonValue* value) {
  if (value == nullptr) {
    return nullptr;
  }
  return std::get_if<std::string>(&value->storage);
}

const std::int64_t* AsInteger(const JsonValue* value) {
  if (value == nullptr) {
    return nullptr;
  }
  return std::get_if<std::int64_t>(&value->storage);
}

bool empty_or_zero_dim(const std::vector<std::size_t>& dims) {
  if (dims.empty()) {
    return true;
  }
  for (std::size_t dim : dims) {
    if (dim == 0) {
      return true;
    }
  }
  return false;
}

void add_issue(
    std::vector<ManifestValidationIssue>* issues,
    ManifestIssueSeverity severity,
    const std::string& tensor_name,
    const std::string& message) {
  issues->push_back(ManifestValidationIssue{
      severity,
      tensor_name,
      message,
  });
}

void add_error(
    std::vector<ManifestValidationIssue>* issues,
    const std::string& tensor_name,
    const std::string& message) {
  add_issue(issues, ManifestIssueSeverity::kError, tensor_name, message);
}

const JsonValue* FindMember(const JsonValue::Object& object, const std::string& key) {
  auto it = object.find(key);
  if (it == object.end()) {
    return nullptr;
  }
  return &it->second;
}

bool ReadRequiredString(
    const JsonValue::Object& object,
    const std::string& key,
    std::string* output,
    std::vector<ManifestValidationIssue>* issues,
    const std::string& tensor_name = "") {
  const JsonValue* value = FindMember(object, key);
  const std::string* string_value = AsString(value);
  if (string_value == nullptr) {
    add_error(issues, tensor_name, "missing or invalid string field '" + key + "'");
    return false;
  }
  *output = *string_value;
  return true;
}

bool ReadOptionalString(
    const JsonValue::Object& object,
    const std::string& key,
    std::string* output) {
  const JsonValue* value = FindMember(object, key);
  const std::string* string_value = AsString(value);
  if (string_value == nullptr) {
    return false;
  }
  *output = *string_value;
  return true;
}

bool ReadRequiredSizeT(
    const JsonValue::Object& object,
    const std::string& key,
    std::size_t* output,
    std::vector<ManifestValidationIssue>* issues,
    const std::string& tensor_name = "") {
  const JsonValue* value = FindMember(object, key);
  const std::int64_t* integer_value = AsInteger(value);
  if (integer_value == nullptr || *integer_value < 0) {
    add_error(issues, tensor_name, "missing or invalid integer field '" + key + "'");
    return false;
  }
  *output = static_cast<std::size_t>(*integer_value);
  return true;
}

bool ReadRequiredDims(
    const JsonValue::Object& object,
    const std::string& key,
    std::vector<std::size_t>* output,
    std::vector<ManifestValidationIssue>* issues,
    const std::string& tensor_name = "") {
  const JsonValue* value = FindMember(object, key);
  const JsonValue::Array* array = AsArray(value);
  if (array == nullptr) {
    add_error(issues, tensor_name, "missing or invalid array field '" + key + "'");
    return false;
  }

  output->clear();
  for (const JsonValue& element : *array) {
    const std::int64_t* integer_value = AsInteger(&element);
    if (integer_value == nullptr || *integer_value <= 0) {
      add_error(issues, tensor_name, "array field '" + key + "' must contain positive integers");
      return false;
    }
    output->push_back(static_cast<std::size_t>(*integer_value));
  }
  return true;
}

std::filesystem::path ResolveManifestPath(
    const std::filesystem::path& manifest_path,
    const std::string& file) {
  const std::filesystem::path file_path(file);
  if (file_path.is_absolute()) {
    return file_path;
  }
  return manifest_path.parent_path() / file_path;
}

std::uint64_t fnv1a_append(std::uint64_t hash, std::uint8_t value) {
  constexpr std::uint64_t kFnvPrime = 1099511628211ull;
  hash ^= value;
  hash *= kFnvPrime;
  return hash;
}

std::string LowercaseCopy(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::optional<std::string> ExtractExpectedFNV64(const std::string& checksum) {
  const std::string lowered = LowercaseCopy(checksum);
  constexpr char kPrefix[] = "fnv1a64:";
  if (lowered.rfind(kPrefix, 0) == 0) {
    return lowered.substr(sizeof(kPrefix) - 1);
  }

  const bool is_plain_hex = lowered.size() == 16 &&
                            std::all_of(
                                lowered.begin(),
                                lowered.end(),
                                [](unsigned char ch) { return std::isxdigit(ch) != 0; });
  if (is_plain_hex) {
    return lowered;
  }
  return std::nullopt;
}

std::optional<std::string> ComputeFNV64ForRange(
    const std::filesystem::path& file_path,
    std::size_t offset_bytes,
    std::size_t nbytes) {
  std::ifstream input(file_path, std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  input.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);
  if (!input) {
    return std::nullopt;
  }

  constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
  std::uint64_t hash = kFnvOffsetBasis;
  std::size_t remaining = nbytes;
  std::string buffer(4096, '\0');

  while (remaining > 0) {
    const std::size_t to_read = std::min(buffer.size(), remaining);
    input.read(buffer.data(), static_cast<std::streamsize>(to_read));
    const std::streamsize read_count = input.gcount();
    if (read_count <= 0) {
      return std::nullopt;
    }
    for (std::streamsize index = 0; index < read_count; ++index) {
      hash = fnv1a_append(hash, static_cast<std::uint8_t>(buffer[static_cast<std::size_t>(index)]));
    }
    remaining -= static_cast<std::size_t>(read_count);
  }

  std::ostringstream oss;
  oss << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << hash;
  return oss.str();
}

}  // namespace

std::vector<ManifestValidationIssue> ValidateManifest(const PackedModelManifest& manifest) {
  std::vector<ManifestValidationIssue> issues;
  if (manifest.schema_version == 0) {
    add_error(&issues, "", "schema_version must be non-zero");
  }
  if (manifest.runtime.model_id.empty()) {
    add_error(&issues, "", "runtime model_id must be non-empty");
  }
  if (manifest.runtime.source_revision.empty()) {
    add_error(&issues, "", "runtime source_revision must be non-empty");
  }
  if (manifest.runtime.tokenizer_revision.empty()) {
    add_error(&issues, "", "runtime tokenizer_revision must be non-empty");
  }
  if (manifest.runtime.kv_bytes_per_token == 0) {
    add_error(&issues, "", "runtime kv_bytes_per_token must be non-zero");
  }
  if (manifest.runtime.mamba_state_bytes_fp16 == 0 ||
      manifest.runtime.mamba_state_bytes_fp32 == 0) {
    add_error(&issues, "", "runtime Mamba state byte sizes must be non-zero for FP16 and FP32");
  }

  std::unordered_set<std::string> tensor_names;
  for (const TensorManifestEntry& tensor : manifest.tensors) {
    if (tensor.name.empty()) {
      add_error(&issues, tensor.name, "tensor name must be non-empty");
    }
    if (!tensor_names.insert(tensor.name).second) {
      add_error(&issues, tensor.name, "tensor name must be unique");
    }
    if (tensor.op_class.empty()) {
      add_error(&issues, tensor.name, "tensor op_class must be non-empty");
    }
    if (empty_or_zero_dim(tensor.logical_shape)) {
      add_error(&issues, tensor.name, "tensor logical_shape must be non-empty with non-zero dimensions");
    }
    if (empty_or_zero_dim(tensor.packed_shape)) {
      add_error(&issues, tensor.name, "tensor packed_shape must be non-empty with non-zero dimensions");
    }
    if (tensor.storage_dtype.empty()) {
      add_error(&issues, tensor.name, "tensor storage_dtype must be non-empty");
    }
    if (tensor.compute_dtype.empty()) {
      add_error(&issues, tensor.name, "tensor compute_dtype must be non-empty");
    }
    if (tensor.layout_tag.empty()) {
      add_error(&issues, tensor.name, "tensor layout_tag must be non-empty");
    }
    if (tensor.alignment_bytes == 0) {
      add_error(&issues, tensor.name, "tensor alignment_bytes must be non-zero");
    }
    if (tensor.packed_file.empty()) {
      add_error(&issues, tensor.name, "tensor packed_file must be non-empty");
    }
    if (tensor.nbytes == 0) {
      add_error(&issues, tensor.name, "tensor nbytes must be non-zero");
    }
    if (tensor.checksum.empty()) {
      add_error(&issues, tensor.name, "tensor checksum must be non-empty");
    }

    const bool has_scale_mode = !tensor.block_scale_mode.empty();
    const bool has_scale_dtype = !tensor.block_scale_dtype.empty();
    const bool has_tensor_scale_dtype = !tensor.tensor_scale_dtype.empty();
    if (has_scale_mode || has_scale_dtype || has_tensor_scale_dtype) {
      if (!(has_scale_mode && has_scale_dtype && has_tensor_scale_dtype)) {
        add_error(&issues, tensor.name, "scaled tensors must provide block_scale_mode, block_scale_dtype, and tensor_scale_dtype together");
      }
      if (tensor.auxiliaries.empty()) {
        add_error(&issues, tensor.name, "scaled tensors must declare auxiliary buffers for scale metadata");
      }
    }

    std::unordered_set<std::string> auxiliary_names;
    for (const TensorAuxiliaryManifest& auxiliary : tensor.auxiliaries) {
      if (auxiliary.name.empty()) {
        add_error(&issues, tensor.name, "auxiliary buffer name must be non-empty");
      }
      if (!auxiliary_names.insert(auxiliary.name).second) {
        add_error(&issues, tensor.name, "auxiliary buffer names must be unique per tensor");
      }
      if (auxiliary.file.empty()) {
        add_error(&issues, tensor.name, "auxiliary buffer file must be non-empty");
      }
      if (auxiliary.nbytes == 0) {
        add_error(&issues, tensor.name, "auxiliary buffer nbytes must be non-zero");
      }
    }
  }

  return issues;
}

bool HasManifestErrors(const std::vector<ManifestValidationIssue>& issues) {
  for (const ManifestValidationIssue& issue : issues) {
    if (issue.severity == ManifestIssueSeverity::kError) {
      return true;
    }
  }
  return false;
}

ManifestLoadResult LoadManifestFromJsonFile(const std::filesystem::path& manifest_path) {
  ManifestLoadResult result;

  std::ifstream input(manifest_path);
  if (!input) {
    add_error(&result.issues, "", "unable to open manifest file: " + manifest_path.string());
    return result;
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();

  JsonValue root;
  try {
    root = JsonParser(buffer.str()).Parse();
  } catch (const std::exception& ex) {
    add_error(&result.issues, "", "manifest JSON parse error: " + std::string(ex.what()));
    return result;
  }

  const JsonValue::Object* root_object = AsObject(&root);
  if (root_object == nullptr) {
    add_error(&result.issues, "", "manifest root must be a JSON object");
    return result;
  }

  PackedModelManifest manifest;
  std::size_t schema_version = 0;
  if (ReadRequiredSizeT(*root_object, "schema_version", &schema_version, &result.issues)) {
    manifest.schema_version = static_cast<std::uint32_t>(schema_version);
  }
  ReadRequiredString(*root_object, "model_id", &manifest.runtime.model_id, &result.issues);
  ReadRequiredString(*root_object, "source_revision", &manifest.runtime.source_revision, &result.issues);
  ReadRequiredString(*root_object, "tokenizer_revision", &manifest.runtime.tokenizer_revision, &result.issues);
  ReadRequiredString(*root_object, "packer_version", &manifest.runtime.packer_version, &result.issues);

  const JsonValue::Object* target_platform = AsObject(FindMember(*root_object, "target_platform"));
  if (target_platform == nullptr) {
    add_error(&result.issues, "", "missing or invalid object field 'target_platform'");
  } else {
    ReadRequiredString(*target_platform, "gpu_family", &manifest.runtime.gpu_family, &result.issues);
    ReadRequiredString(*target_platform, "compute_capability", &manifest.runtime.compute_capability, &result.issues);
  }

  const JsonValue::Object* runtime_profile = AsObject(FindMember(*root_object, "runtime_profile"));
  if (runtime_profile == nullptr) {
    add_error(&result.issues, "", "missing or invalid object field 'runtime_profile'");
  } else {
    ReadRequiredSizeT(*runtime_profile, "kv_bytes_per_token", &manifest.runtime.kv_bytes_per_token, &result.issues);
    ReadRequiredSizeT(*runtime_profile, "mamba_state_bytes_fp16", &manifest.runtime.mamba_state_bytes_fp16, &result.issues);
    ReadRequiredSizeT(*runtime_profile, "mamba_state_bytes_fp32", &manifest.runtime.mamba_state_bytes_fp32, &result.issues);
  }

  const JsonValue::Array* tensors = AsArray(FindMember(*root_object, "tensors"));
  if (tensors == nullptr) {
    add_error(&result.issues, "", "missing or invalid array field 'tensors'");
  } else {
    for (const JsonValue& tensor_value : *tensors) {
      const JsonValue::Object* tensor_object = AsObject(&tensor_value);
      if (tensor_object == nullptr) {
        add_error(&result.issues, "", "tensor entries must be objects");
        continue;
      }

      TensorManifestEntry tensor;
      ReadRequiredString(*tensor_object, "name", &tensor.name, &result.issues);
      ReadRequiredString(*tensor_object, "op_class", &tensor.op_class, &result.issues, tensor.name);
      ReadRequiredDims(*tensor_object, "logical_shape", &tensor.logical_shape, &result.issues, tensor.name);
      ReadRequiredDims(*tensor_object, "packed_shape", &tensor.packed_shape, &result.issues, tensor.name);
      ReadRequiredString(*tensor_object, "storage_dtype", &tensor.storage_dtype, &result.issues, tensor.name);
      ReadRequiredString(*tensor_object, "compute_dtype", &tensor.compute_dtype, &result.issues, tensor.name);
      ReadRequiredString(*tensor_object, "layout_tag", &tensor.layout_tag, &result.issues, tensor.name);
      ReadRequiredSizeT(*tensor_object, "alignment_bytes", &tensor.alignment_bytes, &result.issues, tensor.name);
      ReadOptionalString(*tensor_object, "block_scale_mode", &tensor.block_scale_mode);
      ReadOptionalString(*tensor_object, "block_scale_dtype", &tensor.block_scale_dtype);
      ReadOptionalString(*tensor_object, "tensor_scale_dtype", &tensor.tensor_scale_dtype);
      ReadRequiredString(*tensor_object, "packed_file", &tensor.packed_file, &result.issues, tensor.name);
      ReadRequiredSizeT(*tensor_object, "offset_bytes", &tensor.offset_bytes, &result.issues, tensor.name);
      ReadRequiredSizeT(*tensor_object, "nbytes", &tensor.nbytes, &result.issues, tensor.name);
      if (!ReadOptionalString(*tensor_object, "source_tensor_name", &tensor.source_tensor_name)) {
        ReadOptionalString(*tensor_object, "source_tensor_mapping", &tensor.source_tensor_name);
      }
      ReadRequiredString(*tensor_object, "checksum", &tensor.checksum, &result.issues, tensor.name);

      const JsonValue::Array* auxiliaries = AsArray(FindMember(*tensor_object, "auxiliaries"));
      if (auxiliaries != nullptr) {
        for (const JsonValue& auxiliary_value : *auxiliaries) {
          const JsonValue::Object* auxiliary_object = AsObject(&auxiliary_value);
          if (auxiliary_object == nullptr) {
            add_error(&result.issues, tensor.name, "auxiliary entries must be objects");
            continue;
          }
          TensorAuxiliaryManifest auxiliary;
          ReadRequiredString(*auxiliary_object, "name", &auxiliary.name, &result.issues, tensor.name);
          ReadRequiredString(*auxiliary_object, "file", &auxiliary.file, &result.issues, tensor.name);
          ReadRequiredSizeT(*auxiliary_object, "offset_bytes", &auxiliary.offset_bytes, &result.issues, tensor.name);
          ReadRequiredSizeT(*auxiliary_object, "nbytes", &auxiliary.nbytes, &result.issues, tensor.name);
          tensor.auxiliaries.push_back(std::move(auxiliary));
        }
      }

      manifest.tensors.push_back(std::move(tensor));
    }
  }

  std::vector<ManifestValidationIssue> validation_issues = ValidateManifest(manifest);
  result.issues.insert(result.issues.end(), validation_issues.begin(), validation_issues.end());
  if (!HasManifestErrors(result.issues)) {
    result.ok = true;
    result.manifest = std::move(manifest);
  }
  return result;
}

std::vector<ManifestValidationIssue> VerifyManifestFiles(
    const PackedModelManifest& manifest,
    const std::filesystem::path& manifest_path) {
  std::vector<ManifestValidationIssue> issues;
  for (const TensorManifestEntry& tensor : manifest.tensors) {
    const std::filesystem::path packed_path = ResolveManifestPath(manifest_path, tensor.packed_file);
    std::error_code ec;
    const bool exists = std::filesystem::exists(packed_path, ec);
    if (ec || !exists) {
      add_error(&issues, tensor.name, "packed file not found: " + packed_path.string());
      continue;
    }

    const std::uintmax_t packed_file_size = std::filesystem::file_size(packed_path, ec);
    if (ec) {
      add_error(&issues, tensor.name, "unable to read packed file size: " + packed_path.string());
      continue;
    }
    if (packed_file_size < tensor.offset_bytes + tensor.nbytes) {
      add_error(&issues, tensor.name, "packed file is smaller than required byte range");
    }

    const std::optional<std::string> expected_checksum = ExtractExpectedFNV64(tensor.checksum);
    if (!expected_checksum.has_value()) {
      add_error(&issues, tensor.name, "unsupported checksum format; expected fnv1a64:<hex>");
    } else {
      const std::optional<std::string> actual_checksum =
          ComputeFNV64ForRange(packed_path, tensor.offset_bytes, tensor.nbytes);
      if (!actual_checksum.has_value()) {
        add_error(&issues, tensor.name, "unable to compute checksum over packed tensor bytes");
      } else if (*actual_checksum != *expected_checksum) {
        add_error(&issues, tensor.name, "checksum mismatch for packed tensor bytes");
      }
    }

    for (const TensorAuxiliaryManifest& auxiliary : tensor.auxiliaries) {
      const std::filesystem::path auxiliary_path = ResolveManifestPath(manifest_path, auxiliary.file);
      const bool auxiliary_exists = std::filesystem::exists(auxiliary_path, ec);
      if (ec || !auxiliary_exists) {
        add_error(&issues, tensor.name, "auxiliary file not found: " + auxiliary_path.string());
        continue;
      }
      const std::uintmax_t auxiliary_file_size = std::filesystem::file_size(auxiliary_path, ec);
      if (ec) {
        add_error(&issues, tensor.name, "unable to read auxiliary file size: " + auxiliary_path.string());
        continue;
      }
      if (auxiliary_file_size < auxiliary.offset_bytes + auxiliary.nbytes) {
        add_error(&issues, tensor.name, "auxiliary file is smaller than required byte range");
      }
    }
  }
  return issues;
}

ManifestLoadResult LoadVerifiedManifestFromJsonFile(const std::filesystem::path& manifest_path) {
  ManifestLoadResult result = LoadManifestFromJsonFile(manifest_path);
  if (!result.ok) {
    return result;
  }

  std::vector<ManifestValidationIssue> verification_issues =
      VerifyManifestFiles(result.manifest, manifest_path);
  result.issues.insert(result.issues.end(), verification_issues.begin(), verification_issues.end());
  result.ok = !HasManifestErrors(result.issues);
  return result;
}

}  // namespace nemotron
