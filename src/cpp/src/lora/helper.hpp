// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "openvino/genai/lora_adapter.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// Search for `adapters` property in `properties` map. If it is found and `adapter_config` is not nullptr,
// set `adapter_config` with found value, and return a copy of `properties` with the `adapters` property removed.
// If there is no `adapters` property, `adapter_config` is left unchanged and std::nullopt is returned.
utils::SharedOptional<const AnyMap> extract_adapters_from_properties (const AnyMap& properties, std::optional<AdapterConfig>* adapter_config = nullptr);

// Search for `adapters` property in `properties` map. If it is found, set `adapter_config` with found value and return true.
// If `adapters` property is not found, do nothing and return false.
bool update_adapters_from_properties (const AnyMap& properties, std::optional<AdapterConfig>& adapter_config);

using AdapterConfigAction = std::function<std::optional<AdapterConfig>(const AdapterConfig&)>;
using AdapterAction = std::function<Adapter(const Adapter&)>;

// Update `properties` map with new `adapters` property value. If `properties` map contains `adapters` property,
// call `action` with the value of `adapters` property and update `adapters` property with the result of `action`.
void update_adapters_in_properties(utils::SharedOptional<const AnyMap>& properties, const AdapterConfigAction& action);

// Update `properties` map with new `adapters` property value. If `properties` map contains `adapters` property,
// call `action` with the value of `adapters` property and update `adapters` property with the result of `action`.
utils::SharedOptional<const AnyMap> update_adapters_in_properties(const AnyMap& properties, const AdapterConfigAction& action);

// Create a new AdapterConfig object with adapters modified by `action` function.
std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters, const AdapterAction& action);

// Sets the GenAI-internal part of AdapterConfig: the prefix of the tensors that are nested under tensor_name_prefix
// but are applied by another AdapterController of the same pipeline.
void set_foreign_tensor_name_prefix(AdapterConfig& config, const std::optional<std::string>& prefix);

// Returns true if any LoRA or replacement-constant tensor name matches the configured prefix and predicate.
// The predicate receives the full tensor name, including the prefix.
bool has_lora_tensors(const AdapterConfig& adapters,
					 const std::function<bool(const std::string&)>& predicate);

}
}
