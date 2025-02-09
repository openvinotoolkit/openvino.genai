#include "lora_helper.hpp"


namespace ov {
namespace genai {


std::optional<AnyMap> extract_adapters_from_properties (const AnyMap& properties, std::optional<AdapterConfig>* adapter_config) {
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

bool update_adapters_from_properties (const AnyMap& properties, std::optional<AdapterConfig>& adapter_config) {
    auto adapters_iter = properties.find(AdaptersProperty::name());
    if (adapters_iter != properties.end()) {
        adapter_config = adapters_iter->second.as<AdapterConfig>();
        return true;
    }
    return false;
}

}
}