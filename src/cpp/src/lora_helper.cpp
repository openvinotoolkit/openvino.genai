#include "lora_helper.hpp"


namespace ov {
namespace genai {

std::optional<AnyMap> extract_adapters_from_properties (
        const AnyMap& properties,
        std::optional<AdapterConfig>& adapter_config) {
    auto adapters_iter = properties.find(AdaptersProperty::name());
    if (adapters_iter != properties.end()) {
        adapter_config = std::move(adapters_iter->second.as<AdapterConfig>());
        auto filtered_properties = properties;
        filtered_properties.erase(adapters_iter);
        return filtered_properties;
    }
    return std::nullopt;
}

std::optional<AnyMap> extract_adapters_from_properties (
        const AnyMap& properties,
        AdapterConfig& adapter_config) {
    std::optional<AdapterConfig> adapter_config_tmp;
    if(auto updated_properties = extract_adapters_from_properties(properties, adapter_config_tmp)) {
        adapter_config = std::move(*adapter_config_tmp);
        return updated_properties;
    }
}

}
}