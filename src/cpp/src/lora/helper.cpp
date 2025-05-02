// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "lora/helper.hpp"

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


void update_adapters_in_properties(utils::SharedOptional<const AnyMap>& properties, const AdapterConfigAction& action) {
    std::optional<AdapterConfig> adapter_config;
    if(update_adapters_from_properties(*properties, adapter_config)) {
        if(auto result = action(*adapter_config)) {
            properties.fork()[AdaptersProperty::name()] = *result;
        }
    }
}

utils::SharedOptional<const AnyMap> update_adapters_in_properties(const AnyMap& properties, const AdapterConfigAction& action) {
    utils::SharedOptional<const AnyMap> updated_properties(properties);
    update_adapters_in_properties(updated_properties, action);
    return updated_properties;
}


std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters, const AdapterAction& action) {
    std::optional<AdapterConfig> updated_adapters;
    auto adapters_vector = adapters.get_adapters_and_alphas();
    if(!adapters_vector.empty()) {
        for(auto& adapter: adapters_vector) {
            adapter.first = action(adapter.first);
        }
        updated_adapters = AdapterConfig(adapters_vector, adapters.get_mode());
        updated_adapters->set_tensor_name_prefix(adapters.get_tensor_name_prefix());
    }
    return updated_adapters;
}

}
}