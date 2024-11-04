#pragma once

#include <string>
#include <unordered_map>
#include <set>

namespace ov {
namespace genai {

using NameMap = std::unordered_map<std::string, std::string>;

NameMap maybe_map_sgm_blocks_to_diffusers(std::set<std::string> state_dict, int layers_per_block = 2,
                                           const std::string& delimiter = "_", int block_slice_pos = 5);

NameMap maybe_map_non_diffusers_lora_to_diffusers(const std::set<std::string>& keys);

}
}
