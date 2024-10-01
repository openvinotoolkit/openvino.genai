#pragma once

#include <optional>

#include "openvino/genai/lora_adapter.hpp"


namespace ov {
namespace genai {

// Search for `adapters` property in `properties` map. If it is found, set adapter_config with found value and
// return a copy of properties with the `adapter` removed. If there is no `adapters` property, `adapter_config` is
// left unchanged and std::nullopt is returned.
std::optional<AnyMap> extract_adapters_from_properties (const AnyMap& properties, AdapterConfig& adapter_config);

// Doing the same as `extract_adapters_from_properties` but not returning updated properties
void get_adapters_from_properties (const AnyMap& properties, AdapterConfig& adapter_config);

std::optional<AnyMap> filter_out_adapters_from_properties (const AnyMap& properties);

}
}