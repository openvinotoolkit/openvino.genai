#pragma once

#include <string>
#include <unordered_map>
#include <set>
#include <regex>

#include <openvino/genai/lora_adapter.hpp>

#include "lora/common.hpp"

namespace ov {
namespace genai {
namespace utils {

using NameMap = std::unordered_map<std::string, std::string>;

// Holds a compiled regex pattern and an index to a particular capture group
// operator() takes a string, parses it with that regex pattern and returns the capture group value
struct RegexParser {
    std::regex pattern;
    std::vector<size_t> capture_indices;
    RegexParser (const std::string& pattern, size_t capture_index) : pattern(pattern), capture_indices(1, capture_index) {}
    RegexParser (const std::string& pattern, const std::vector<size_t>& capture_indices) : pattern(pattern), capture_indices(capture_indices) {}
    std::optional<std::string> operator() (const std::string& name) const;
};

NameMap maybe_map_sgm_blocks_to_diffusers(std::set<std::string> state_dict, int layers_per_block = 2,
                                           const std::string& delimiter = "_", int block_slice_pos = 5);

NameMap maybe_map_non_diffusers_lora_to_diffusers(const std::set<std::string>& keys);

void convert_prefix_te(std::string& name);

utils::LoRATensors flux_kohya_lora_preprocessing(const utils::LoRATensors& tensors);

utils::LoRATensors flux_xlabs_lora_preprocessing(const utils::LoRATensors& tensors);

}

Adapter flux_adapter_normalization(const Adapter& adapter);

Adapter diffusers_adapter_normalization(const Adapter& adapter);

}
}
