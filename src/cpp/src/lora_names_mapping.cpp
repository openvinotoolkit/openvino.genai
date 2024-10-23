// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Content of this file is a C++ port of the name mapping for LoRA tensors from HuggingFace diffusers/loaders/lora_conversion_utils.py
// Implementation doesn't exactly match because we are doing suffix processing of LoRA tensors in another place.

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include <stdexcept>
#include <regex>
#include <algorithm>

#include "lora_names_mapping.hpp"

namespace {

using BlocksMap = std::unordered_map<int, std::vector<std::string>>;

// Helper function to split a string by a delimiter
std::vector<std::string> split(const std::string &s, const std::string &delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = s.find(delimiter);  // Find the first occurrence of delimiter

    while (end != std::string::npos) {
        result.push_back(s.substr(start, end - start));  // Extract substring
        start = end + delimiter.length();  // Move the start position
        end = s.find(delimiter, start);  // Find the next occurrence
    }

    // Add the last token after the final delimiter
    result.push_back(s.substr(start));
    return result;
}


// Helper function to join a vector of strings with a delimiter
std::string join(const std::vector<std::string>& parts, char delimiter) {
    std::string result;
    for(size_t i = 0; i < parts.size(); ++i){
        result += parts[i];
        if(i != parts.size() -1){
            result += delimiter;
        }
    }
    return result;
}


std::string _convert_unet_lora_key(const std::string& key) {
    std::string diffusers_name = key;

    diffusers_name = std::regex_replace(diffusers_name, std::regex("lora.unet"), "lora_unet");

    if(key.find("lora_unet") != 0) {
        return key;
    }

    diffusers_name = std::regex_replace(diffusers_name, std::regex("_"), ".");

    diffusers_name = std::regex_replace(diffusers_name, std::regex("input\\.blocks"), "down_blocks");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("down\\.blocks"), "down_blocks");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("middle\\.block"), "mid_block");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("mid\\.block"), "mid_block");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("output\\.blocks"), "up_blocks");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("up\\.blocks"), "up_blocks");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("transformer\\.blocks"), "transformer_blocks");
    // Original patterns in HF are different for the next block, because 'lora' suffix is already processed
    diffusers_name = std::regex_replace(diffusers_name, std::regex("to\\.q"), "to_q");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("to\\.k"), "to_k");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("to\\.v"), "to_v");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("to\\.out\\.0"), "to_out");

    diffusers_name = std::regex_replace(diffusers_name, std::regex("proj\\.in"), "proj_in");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("proj\\.out"), "proj_out");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("emb\\.layers"), "time_emb_proj");

    // Regex match for SDXL specific conversions
    if (diffusers_name.find("emb") != std::string::npos && diffusers_name.find("time.emb.proj") == std::string::npos) {
        diffusers_name = std::regex_replace(diffusers_name, std::regex("\\.\\d+(?=\\D*$)"), "");
    }

    if (diffusers_name.find(".in.") != std::string::npos) {
        diffusers_name = std::regex_replace(diffusers_name, std::regex("in\\.layers\\.2"), "conv1");
    }

    if (diffusers_name.find(".out.") != std::string::npos) {
        diffusers_name = std::regex_replace(diffusers_name, std::regex("out\\.layers\\.3"), "conv2");
    }

    if (diffusers_name.find("downsamplers") != std::string::npos || diffusers_name.find("upsamplers") != std::string::npos) {
        diffusers_name = std::regex_replace(diffusers_name, std::regex("op"), "conv");
    }

    if (diffusers_name.find("skip") != std::string::npos) {
        diffusers_name = std::regex_replace(diffusers_name, std::regex("skip\\.connection"), "conv_shortcut");
    }

    diffusers_name = std::regex_replace(diffusers_name, std::regex("lora.unet"), "lora_unet");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("lora.te"), "lora_te");
    diffusers_name = std::regex_replace(diffusers_name, std::regex("base.model"), "base_model");

    return diffusers_name;
}

}


namespace ov {
namespace genai {

// Function to reimplement _maybe_map_sgm_blocks_to_diffusers
NameMap maybe_map_sgm_blocks_to_diffusers(std::set<std::string> state_dict, int layers_per_block,
                                           const std::string& delimiter, int block_slice_pos) {
    // 1. Get all state_dict keys
    std::vector<std::string> all_keys;
    for (const auto& key : state_dict) {
        all_keys.push_back(key);
    }

    std::vector<std::string> sgm_patterns = {"input_blocks", "middle_block", "output_blocks"};

    // 2. Check if needs remapping, if not return original dict
    bool is_in_sgm_format = false;
    for (const auto& key : all_keys) {
        for (const auto& pattern : sgm_patterns) {
            if (key.find(pattern) != std::string::npos) {
                is_in_sgm_format = true;
                break;
            }
        }
        if (is_in_sgm_format) break;
    }

    if (!is_in_sgm_format) {
        return NameMap{};
    }

    // 3. Else remap from SGM patterns
    NameMap new_state_dict;
    std::vector<std::string> inner_block_map = {"resnets", "attentions", "upsamplers"};

    // Retrieve number of down, mid and up blocks
    std::set<int> input_block_ids, middle_block_ids, output_block_ids;

    for (const auto& layer : all_keys) {
        if (layer.find("text") != std::string::npos) {
            // Pass, no mapping for this layer
            state_dict.erase(layer);
        } else {
            // Split the layer by delimiter and get the layer_id
            std::vector<std::string> parts = split(layer, delimiter);
            if (parts.size() < block_slice_pos) {
                throw std::runtime_error("Layer format is incorrect: " + layer);
            }
            int layer_id = std::stoi(parts[block_slice_pos -1]); // zero-based indexing

            if (layer.find("input_blocks") != std::string::npos) {
                input_block_ids.insert(layer_id);
            }
            else if (layer.find("middle_block") != std::string::npos) {
                middle_block_ids.insert(layer_id);
            }
            else if (layer.find("output_blocks") != std::string::npos) {
                output_block_ids.insert(layer_id);
            }
            else {
                throw std::runtime_error("Checkpoint not supported because layer " + layer + " not supported.");
            }
        }
    }

    // Function to collect keys based on block IDs and pattern
    auto collect_blocks = [&](const std::set<int>& block_ids, const std::string& pattern) -> BlocksMap {
        BlocksMap blocks;
        for (const auto& id : block_ids) {
            std::vector<std::string> matched_keys;
            for (const auto& key : state_dict) {
                if (key.find(pattern + delimiter + std::to_string(id)) != std::string::npos) {
                    matched_keys.push_back(key);
                }
            }
            blocks[id] = matched_keys;
        }
        return blocks;
    };

    BlocksMap input_blocks = collect_blocks(input_block_ids, sgm_patterns[0]);
    BlocksMap middle_blocks = collect_blocks(middle_block_ids, sgm_patterns[1]);
    BlocksMap output_blocks = collect_blocks(output_block_ids, sgm_patterns[2]);

    // Rename keys accordingly
    // Handle Input Blocks
    for (const auto& [i, keys] : input_blocks) {
        int block_id = (i -1) / (layers_per_block +1);
        int layer_in_block_id = (i -1) % (layers_per_block +1);

        for (const auto& key : keys) {
            // Split key
            std::vector<std::string> parts = split(key, delimiter);
            if (parts.size() < block_slice_pos) {
                throw std::runtime_error("Layer format is incorrect: " + key);
            }
            int inner_block_id = std::stoi(parts[block_slice_pos]);
            std::string inner_block_key = (key.find("op") == std::string::npos) ? inner_block_map[inner_block_id] : "downsamplers";
            std::string inner_layers_in_block = (key.find("op") == std::string::npos) ? std::to_string(layer_in_block_id) : "0";

            // Create new key
            std::vector<std::string> new_parts;
            // Copy parts up to block_slice_pos -1
            for(int idx =0; idx < block_slice_pos -1; ++idx){
                new_parts.push_back(parts[idx]);
            }
            // Add new parts
            new_parts.push_back(std::to_string(block_id));
            new_parts.push_back(inner_block_key);
            new_parts.push_back(inner_layers_in_block);
            // Add remaining parts after block_slice_pos
            for(int idx = block_slice_pos +1; idx < parts.size(); ++idx){
                new_parts.push_back(parts[idx]);
            }

            std::string new_key = join(new_parts, delimiter[0]);
            new_state_dict[key] = new_key;
            state_dict.erase(key);
        }
    }

    // Handle Middle Blocks
    for (const auto& [i, keys] : middle_blocks) {
        std::vector<std::string> key_part;
        if (i == 0) {
            key_part = {inner_block_map[0], "0"};
        }
        else if (i == 1) {
            key_part = {inner_block_map[1], "0"};
        }
        else if (i == 2) {
            key_part = {inner_block_map[0], "1"};
        }
        else {
            throw std::runtime_error("Invalid middle block id " + std::to_string(i));
        }

        for (const auto& key : keys) {
            // Split key
            std::vector<std::string> parts = split(key, delimiter);
            if (parts.size() < block_slice_pos) {
                throw std::runtime_error("Layer format is incorrect: " + key);
            }

            // Create new key
            std::vector<std::string> new_parts;
            // Copy parts up to block_slice_pos -1
            for(int idx =0; idx < block_slice_pos -1; ++idx){
                new_parts.push_back(parts[idx]);
            }
            // Add key_part
            new_parts.insert(new_parts.end(), key_part.begin(), key_part.end());
            // Add remaining parts after block_slice_pos
            for(int idx = block_slice_pos; idx < parts.size(); ++idx){
                new_parts.push_back(parts[idx]);
            }

            std::string new_key = join(new_parts, delimiter[0]);
            new_state_dict[key] = new_key;
            state_dict.erase(key);
        }
    }

    // Handle Output Blocks
    for (const auto& [i, keys] : output_blocks) {
        int block_id = i / (layers_per_block +1);
        int layer_in_block_id = i % (layers_per_block +1);

        for (const auto& key : keys) {
            // Split key
            std::vector<std::string> parts = split(key, delimiter);
            if (parts.size() < block_slice_pos) {
                throw std::runtime_error("Layer format is incorrect: " + key);
            }
            int inner_block_id = std::stoi(parts[block_slice_pos]);
            std::string inner_block_key = inner_block_map[inner_block_id];
            std::string inner_layers_in_block = (inner_block_id < 2) ? std::to_string(layer_in_block_id) : "0";

            // Create new key
            std::vector<std::string> new_parts;
            // Copy parts up to block_slice_pos -1
            for(int idx =0; idx < block_slice_pos -1; ++idx){
                new_parts.push_back(parts[idx]);
            }
            // Add new parts
            new_parts.push_back(std::to_string(block_id));
            new_parts.push_back(inner_block_key);
            new_parts.push_back(inner_layers_in_block);
            // Add remaining parts after block_slice_pos
            for(int idx = block_slice_pos +1; idx < parts.size(); ++idx){
                new_parts.push_back(parts[idx]);
            }

            std::string new_key = join(new_parts, delimiter[0]);
            new_state_dict[key] = new_key;
            state_dict.erase(key);
        }
    }

    // After remapping, ensure all keys have been converted
    if (!state_dict.empty()) {
        std::string remaining_keys;
        for (const auto& key : state_dict) {
            remaining_keys += key + ", ";
        }
        throw std::runtime_error("At this point all state dict entries have to be converted. Remaining keys: " + remaining_keys);
    }

    return new_state_dict;
}


NameMap maybe_map_non_diffusers_lora_to_diffusers(const std::set<std::string>& keys) {
    NameMap new_keys = maybe_map_sgm_blocks_to_diffusers(keys);
    for(const auto& key: keys) {
        std::string new_key = key;
        auto it = new_keys.find(new_key);
        if(new_keys.end() != it) {
            new_key = it->second;
        }
        new_key = _convert_unet_lora_key(new_key);
        new_keys[key] = new_key;
    }
    return new_keys;
}


}
}