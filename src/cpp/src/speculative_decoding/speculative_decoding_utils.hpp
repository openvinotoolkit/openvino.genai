// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "openvino/runtime/core.hpp"

namespace ov {
namespace genai {
namespace speculative_decoding {

/**
 * @brief Eagle3 runtime configuration information
 */
struct Eagle3RTInfo {
    bool eagle3_mode = false;
    std::vector<int> hidden_layers_list;
    std::filesystem::path dt_mapping_table;
};

/**
 * @brief Extract Eagle3 configuration from draft model properties
 *
 * This function extracts Eagle3-specific configuration from the draft model's
 * property map. It looks for:
 * - eagle3_mode: boolean flag to enable Eagle3 speculative decoding
 * - hidden_layers_list: explicit list of layer indices to extract hidden states from
 *
 * If hidden_layers_list is not provided and models_path is given, the function
 * will attempt to auto-deduce the layers from the model's config.json file.
 *
 * @param config Draft model configuration map (will be modified - eagle3 params will be erased)
 * @param models_path Optional path to model directory for auto-deducing hidden layers from config.json
 * @return Eagle3RTInfo structure with extracted configuration
 */
Eagle3RTInfo extract_eagle_mode_from_config(ov::AnyMap& config, const std::filesystem::path& models_path = {});

}  // namespace speculative_decoding
}  // namespace genai
}  // namespace ov
