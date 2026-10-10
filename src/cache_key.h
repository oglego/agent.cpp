#pragma once

// Cache compatibility key for prompt cache files.
//
// llama_state_save_file() stores the KV cache but nothing about the adapters
// (or other settings) that produced it. This module records those settings in
// a small sidecar file next to the cache (<cache>.meta) so a mismatched cache
// can be detected and rejected instead of silently reused.
//
// Deliberately free of llama.cpp types so it can be unit tested standalone.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace agent_cpp {

/// Plain inputs for building a CacheKey (no llama.cpp types).
struct CacheKeyInputs
{
    struct Lora
    {
        std::string path;
        float scale = 1.0F;
        std::string file_identity;
    };

    std::string model_identity;
    int n_ctx = 0;
    int cache_type_k = 0;
    int cache_type_v = 0;
    std::vector<Lora> loras;
};

/// Ordered key/value description of everything that makes a cache reusable.
/// Stored as individual fields (not one opaque hash) so a mismatch can say
/// which setting changed.
class CacheKey
{
  public:
    static constexpr const char* FORMAT_VERSION = "2";

    /// Build the key. Adapters are LoRA deltas that add, so their order does
    /// not matter; entries are sorted to make the key order-independent.
    /// Adapters with scale 0 are ignored (llama.cpp never applies them).
    static CacheKey from_inputs(const CacheKeyInputs& in)
    {
        CacheKey key;
        key.fields_["format"] = FORMAT_VERSION;
        key.fields_["model"] = in.model_identity;
        key.fields_["n_ctx"] = std::to_string(in.n_ctx);
        key.fields_["cache_type_k"] = std::to_string(in.cache_type_k);
        key.fields_["cache_type_v"] = std::to_string(in.cache_type_v);

        std::vector<std::string> adapters;
        for (const auto& lora : in.loras) {
            if (lora.scale == 0.0F) {
                continue;
            }
            adapters.push_back(describe_adapter(
              lora.path, lora.scale, lora.file_identity));
        }
        std::sort(adapters.begin(), adapters.end());

        std::string joined;
        for (const auto& a : adapters) {
            joined += (joined.empty() ? "" : ";") + a;
        }
        key.fields_["adapters"] = joined;
        return key;
    }

    /// Path of the sidecar for a given cache file.
    static std::string sidecar_path(const std::string& cache_path)
    {
        return cache_path + ".meta";
    }

    /// Empty string if compatible, otherwise a human-readable reason.
    [[nodiscard]] std::string mismatch(const CacheKey& stored) const
    {
        for (const auto& [name, value] : fields_) {
            auto it = stored.fields_.find(name);
            if (it == stored.fields_.end()) {
                return "field '" + name + "' missing from stored metadata";
            }
            if (it->second != value) {
                return "'" + name + "' changed (cache: " + it->second +
                       ", current: " + value + ")";
            }
        }
        for (const auto& [name, value] : stored.fields_) {
            if (fields_.find(name) == fields_.end()) {
                return "unexpected field '" + name + "' in stored metadata";
            }
        }
        return {};
    }

    [[nodiscard]] std::string serialize() const
    {
        std::ostringstream out;
        for (const auto& [name, value] : fields_) {
            out << name << '=' << value << '\n';
        }
        return out.str();
    }

    /// Returns false if the text is not a valid key.
    static bool parse(const std::string& text, CacheKey& out)
    {
        CacheKey key;
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            auto eq = line.find('=');
            if (eq == std::string::npos || eq == 0) {
                return false;
            }
            key.fields_[line.substr(0, eq)] = line.substr(eq + 1);
        }
        if (key.fields_.empty()) {
            return false;
        }
        out = std::move(key);
        return true;
    }

    /// Atomically write the sidecar (temp file + rename).
    bool write(const std::string& cache_path) const
    {
        const std::string final_path = sidecar_path(cache_path);
        const std::string tmp_path = final_path + ".tmp";
        {
            std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
            if (!file) {
                return false;
            }
            file << serialize();
            file.flush();
            if (!file) {
                return false;
            }
        }
        std::error_code ec;
        std::filesystem::rename(tmp_path, final_path, ec);
        if (ec) {
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
        return true;
    }

    /// Returns false if the sidecar is missing or unparseable.
    static bool read(const std::string& cache_path, CacheKey& out)
    {
        std::ifstream file(sidecar_path(cache_path), std::ios::binary);
        if (!file) {
            return false;
        }
        std::ostringstream contents;
        contents << file.rdbuf();
        return parse(contents.str(), out);
    }

    /// Remove any existing sidecar (call before overwriting a cache so a
    /// crash mid-save leaves "no sidecar", which is rejected on load).
    static void remove(const std::string& cache_path)
    {
        std::error_code ec;
        std::filesystem::remove(sidecar_path(cache_path), ec);
    }

    [[nodiscard]] const std::map<std::string, std::string>& fields() const
    {
        return fields_;
    }

    /// Capture a file's path, size, and modification time for stable cache
    /// identity. Call this when the file is loaded, then retain the result.
    static std::string file_identity(const std::string& path)
    {
        std::error_code ec;
        std::filesystem::path canon =
          std::filesystem::weakly_canonical(path, ec);
        const std::string canon_str = ec ? path : canon.string();

        std::error_code size_ec;
        std::error_code time_ec;
        const auto size = std::filesystem::file_size(path, size_ec);
        const auto mtime = std::filesystem::last_write_time(path, time_ec);

        std::ostringstream ident;
        ident << canon_str << '|';
        if (size_ec || time_ec) {
            ident << "unreadable";
        } else {
            ident << size << '|'
                  << static_cast<std::int64_t>(
                       mtime.time_since_epoch().count());
        }
        return ident.str();
    }

  private:
    static uint64_t fnv1a(const std::string& s)
    {
        uint64_t h = 1469598103934665603ULL;
        for (unsigned char c : s) {
            h ^= c;
            h *= 1099511628211ULL;
        }
        return h;
    }

    // "<hash of file identity>@<scale>". The identity is captured when the
    // adapter is loaded so later file changes cannot relabel the loaded state.
    static std::string describe_adapter(const std::string& path,
                                        float scale,
                                        const std::string& file_identity)
    {
        const std::string identity =
          file_identity.empty() ? CacheKey::file_identity(path) : file_identity;

        char buf[64];
        std::snprintf(buf,
                      sizeof(buf),
                      "%016llx@%.9g",
                      static_cast<unsigned long long>(fnv1a(identity)),
                      static_cast<double>(scale));
        return buf;
    }

    std::map<std::string, std::string> fields_;
};

} // namespace agent_cpp
