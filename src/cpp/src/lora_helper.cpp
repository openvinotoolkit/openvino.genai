#include "lora_helper.hpp"


namespace ov {
namespace genai {


utils::SharedOptional<const AnyMap> extract_adapters_from_properties (const AnyMap& properties, std::optional<AdapterConfig>* adapter_config) {
    auto adapters_iter = properties.find(AdaptersProperty::name());
    utils::SharedOptional<const AnyMap> filtered_properties(&properties);
    if (adapters_iter != properties.end()) {
        if(adapter_config) {
            *adapter_config = adapters_iter->second.as<AdapterConfig>();
        }
        filtered_properties.fork().erase(AdaptersProperty::name());
    }
    return filtered_properties;
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