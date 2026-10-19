#pragma once

#include "model.h"
#include <iomanip>
#include <sstream>

namespace agent_cpp::detail {

inline std::string
cache_metadata(const std::string& model_fingerprint,
               const std::vector<LoraAdapterConfig>& adapters,
               const std::vector<std::string>& adapter_fingerprints)
{
    if (model_fingerprint.empty() ||
        adapters.size() != adapter_fingerprints.size()) {
        return {};
    }

    std::ostringstream result;
    result << "agent-cpp-cache-v2\n" << model_fingerprint << '\n'
           << adapters.size() << '\n';
    for (size_t i = 0; i < adapters.size(); ++i) {
        if (adapter_fingerprints[i].empty()) {
            return {};
        }
        result << adapter_fingerprints[i] << ' ' << std::hexfloat
               << adapters[i].scale << '\n';
    }
    return result.str();
}

} // namespace agent_cpp::detail
