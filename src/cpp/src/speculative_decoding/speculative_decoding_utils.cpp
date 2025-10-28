// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_utils.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"

namespace ov {
namespace genai {
namespace speculative_decoding {

Eagle3RTInfo extract_eagle_mode_from_config(ov::AnyMap& config, const std::filesystem::path& models_path) {
    Eagle3RTInfo eagle_rt_info;

    // Check if eagle3_mode is enabled
    if (config.find("eagle3_mode") != config.end()) {
        eagle_rt_info.eagle3_mode = config.at("eagle3_mode").as<bool>();
        config.erase("eagle3_mode");

        // Try to get explicit hidden_layers_list from config
        if (config.find("hidden_layers_list") != config.end()) {
            eagle_rt_info.hidden_layers_list = config.at("hidden_layers_list").as<std::vector<int>>();
            config.erase("hidden_layers_list");
        }
        // Auto-deduce from config.json if models_path is provided
        else if (!models_path.empty()) {
            auto config_file_path = models_path / "config.json";
            if (std::filesystem::exists(config_file_path)) {
                std::ifstream file(config_file_path);
                nlohmann::json data = nlohmann::json::parse(file);

                using ov::genai::utils::read_json_param;
                int num_decoder_layers = 0;
                read_json_param(data, "num_hidden_layers", num_decoder_layers);

                if (num_decoder_layers > 3) {
                    // Auto-deduce hidden layers: early, middle, late layers
                    // Formula: {2, num_layers/2, num_layers-3}
                    // Example: 32 layers -> {2, 16, 29}
                    eagle_rt_info.hidden_layers_list = {2, num_decoder_layers / 2, num_decoder_layers - 3};
                }
            }
        }
    }

    return eagle_rt_info;
}

}  // namespace speculative_decoding
}  // namespace genai
}  // namespace ov
