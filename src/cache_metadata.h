#pragma once

#include "model.h"
#include <iomanip>
#include <sstream>

namespace agent_cpp::detail {

inline std::string
cache_metadata(const std::string& model_fingerprint,
               const ModelConfig& config,
               const std::vector<std::string>& adapter_fingerprints)
{
    if (model_fingerprint.empty() ||
        config.loras.size() != adapter_fingerprints.size()) {
        return {};
    }

    std::ostringstream result;
    result << "agent-cpp-cache-v3\n" << model_fingerprint << '\n'
           << config.n_ctx << ' ' << static_cast<int>(config.cache_type_k)
           << ' ' << static_cast<int>(config.cache_type_v) << '\n'
           << config.loras.size() << '\n';
    for (size_t i = 0; i < config.loras.size(); ++i) {
        if (adapter_fingerprints[i].empty()) {
            return {};
        }
        result << adapter_fingerprints[i] << ' ' << std::hexfloat
               << config.loras[i].scale << '\n';
    }
    return result.str();
}

} // namespace agent_cpp::detail
