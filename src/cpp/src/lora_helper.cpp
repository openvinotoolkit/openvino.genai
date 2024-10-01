#include "lora_helper.hpp"

namespace {

using namespace ov;
using namespace ov::genai;

std::optional<AnyMap> extract_adapters_from_properties (const AnyMap& properties, AdapterConfig* adapter_config = nullptr) {
    auto adapters_iter = properties.find(AdaptersProperty::name());
    if (adapters_iter != properties.end()) {
        if(adapter_config) {
            *adapter_config = adapters_iter->second.as<AdapterConfig>();
        }
        auto filtered_properties = properties;
        filtered_properties.erase(AdaptersProperty::name());
        return filtered_properties;
    }
    return std::nullopt;
}

}

namespace ov {
namespace genai {

std::optional<AnyMap> extract_adapters_from_properties (const AnyMap& properties, AdapterConfig& adapter_config) {
    return ::extract_adapters_from_properties(properties, &adapter_config);
}

void get_adapters_from_properties (const AnyMap& properties, AdapterConfig& adapter_config) {
    auto adapters_iter = properties.find(AdaptersProperty::name());
    if (adapters_iter != properties.end()) {
        adapter_config = adapters_iter->second.as<AdapterConfig>();
    }
}

std::optional<AnyMap> filter_out_adapters_from_properties (const AnyMap& properties) {
    return ::extract_adapters_from_properties(properties);
}

}
}