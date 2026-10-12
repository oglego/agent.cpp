#pragma once

// Identity of a prompt (KV) cache file.
//
// The cache file's name carries a digest of everything that decides whether
// its contents are valid for a Model, so there is no sidecar to keep in sync:
// a mismatched cache is simply a different file, i.e. a miss.
//
//   digest(base model, LoRA adapters (content, scale), context params)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace agent_cpp {

/// 64-bit FNV-1a. Not cryptographic: it names local cache files, nothing more.
class Fnv1a
{
  public:
    void update(const void* data, size_t len)
    {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < len; ++i) {
            h_ = (h_ ^ p[i]) * 1099511628211ULL;
        }
    }

    // Fixed width, little-endian: fields can't run together
    void update_u64(uint64_t v)
    {
        uint8_t b[8];
        for (int i = 0; i < 8; ++i) {
            b[i] = static_cast<uint8_t>(v >> (8 * i));
        }
        update(b, sizeof(b));
    }

    [[nodiscard]] uint64_t value() const { return h_; }

  private:
    uint64_t h_ = 14695981039346656037ULL;
};

/// Hash of a file's contents (for small files such as LoRA adapters).
/// nullopt if it can't be read.
inline std::optional<uint64_t>
hash_file(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    Fnv1a h;
    std::vector<char> buf(1 << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        h.update(buf.data(), static_cast<size_t>(in.gcount()));
    }
    if (in.bad()) {
        return std::nullopt;
    }
    return h.value();
}

/// Cheap identity for large files (the base model): name, size and mtime.
/// nullopt if the file can't be inspected.
inline std::optional<uint64_t>
file_identity(const std::filesystem::path& path)
{
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        return std::nullopt;
    }
    const auto mtime = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return std::nullopt;
    }
    const std::string name = path.filename().string();
    Fnv1a h;
    h.update_u64(name.size());
    h.update(name.data(), name.size());
    h.update_u64(size);
    h.update_u64(static_cast<uint64_t>(mtime.time_since_epoch().count()));
    return h.value();
}

struct CacheContext
{
    int n_ctx = 0;
    int cache_type_k = 0;
    int cache_type_v = 0;
    int session_version = 0; // LLAMA_SESSION_VERSION: KV state file format
};

/// Digest of a base model plus the adapters stacked on it. Independent of
/// adapter order; adapters with scale 0 contribute nothing (they don't change
/// the output, so they must not split the cache).
inline uint64_t
cache_digest(uint64_t model_id,
             std::vector<std::pair<uint64_t, float>> adapters,
             const CacheContext& ctx)
{
    adapters.erase(std::remove_if(adapters.begin(),
                                  adapters.end(),
                                  [](const auto& a) { return a.second == 0.0F; }),
                   adapters.end());

    std::vector<std::pair<uint64_t, uint32_t>> keyed; // scale as raw bits
    keyed.reserve(adapters.size());
    for (const auto& [hash, scale] : adapters) {
        uint32_t bits = 0;
        std::memcpy(&bits, &scale, sizeof(bits));
        keyed.emplace_back(hash, bits);
    }
    std::sort(keyed.begin(), keyed.end());

    Fnv1a h;
    h.update_u64(model_id);
    h.update_u64(keyed.size());
    for (const auto& [hash, bits] : keyed) {
        h.update_u64(hash);
        h.update_u64(bits);
    }
    h.update_u64(static_cast<uint64_t>(ctx.n_ctx));
    h.update_u64(static_cast<uint64_t>(ctx.cache_type_k));
    h.update_u64(static_cast<uint64_t>(ctx.cache_type_v));
    h.update_u64(static_cast<uint64_t>(ctx.session_version));
    return h.value();
}

/// "dir/foo.bin" + digest -> "dir/foo.<16 hex chars>.bin"
inline std::filesystem::path
keyed_cache_path(const std::filesystem::path& base, uint64_t digest)
{
    static constexpr char kHex[] = "0123456789abcdef";
    std::string hex(16, '0');
    for (int i = 15; i >= 0; --i, digest >>= 4) {
        hex[i] = kHex[digest & 0xF];
    }
    return base.parent_path() /
           (base.stem().string() + "." + hex + base.extension().string());
}

} // namespace agent_cpp
