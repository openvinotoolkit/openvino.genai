#pragma once

#include <optional>

#include "openvino/genai/lora_adapter.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// Search for `adapters` property in `properties` map. If it is found and `adapter_config` is not nullptr,
// set `adapter_config` with found value, and return a copy of `properties` with the `adapters` property removed.
// If there is no `adapters` property, `adapter_config` is left unchanged and std::nullopt is returned.
std::optional<AnyMap> extract_adapters_from_properties (const AnyMap& properties, std::optional<AdapterConfig>* adapter_config = nullptr);

// Search for `adapters` property in `properties` map. If it is found, set `adapter_config` with found value and return true.
// If `adapters` property is not found, do nothing and return false.
bool update_adapters_from_properties (const AnyMap& properties, std::optional<AdapterConfig>& adapter_config);


template <typename OptionalAction>
utils::DefaultOptional<const AnyMap> update_adapters_in_properties(const AnyMap& properties, const OptionalAction& action) {
    std::optional<AdapterConfig> adapter_config;
    if(update_adapters_from_properties(properties, adapter_config)) {
        if(auto result = action(*adapter_config)) {
            AnyMap updated_properties = properties;
            updated_properties[AdaptersProperty::name()] = *result;
            return utils::DefaultOptional<const AnyMap>(updated_properties);
        }
    }
    return utils::DefaultOptional<const AnyMap>(&properties);
}

template <typename OptionalAction>
std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters, const OptionalAction& action) {
    std::optional<AdapterConfig> updated_adapters;
    const auto& adapters_vector = adapters.get_adapters();
    if(!adapters_vector.empty()) {
        updated_adapters = adapters;    // it is simpler w.r.t coding to just copy the config entirely with all modes/options and adapters, and replace adapters later
        for(const auto& adapter: adapters_vector) {
            updated_adapters->remove(adapter);
            updated_adapters->add(action(adapter), adapters.get_alpha(adapter));
        }
    }
    return updated_adapters;
}

}
}