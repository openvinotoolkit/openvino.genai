#pragma once

#include <optional>

#include "openvino/genai/lora_adapter.hpp"


namespace ov {
namespace genai {

// Search for `adapters` property in `properties` map. If it is found and `adapter_config` is not nullptr,
// set `adapter_config` with found value, and return a copy of `properties` with the `adapters` property removed.
// If there is no `adapters` property, `adapter_config` is left unchanged and std::nullopt is returned.
std::optional<AnyMap> extract_adapters_from_properties (const AnyMap& properties, AdapterConfig* adapter_config = nullptr);

// Search for `adapters` property in `properties` map. If it is found, set `adapter_config` with found value.
// If `adapters` property is not found, do nothing.
void update_adapters_from_properties (const AnyMap& properties, AdapterConfig& adapter_config);

}
}