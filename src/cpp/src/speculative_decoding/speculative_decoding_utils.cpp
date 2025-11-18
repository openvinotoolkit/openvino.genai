// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_utils.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"

namespace ov {
namespace genai {
namespace speculative_decoding {

void ensure_num_assistant_tokens_is_set(ov::genai::GenerationConfig& config) {
    // Only num_assistant_tokens is supported, not assistant_confidence_threshold
    OPENVINO_ASSERT(
        config.assistant_confidence_threshold == 0.f,
        "Speculative Decoding only supports num_assistant_tokens, not assistant_confidence_threshold. Set it to 0.f.");
    if (config.num_assistant_tokens == 0) {
        config.num_assistant_tokens = DEFAULT_NUM_ASSISTANT_TOKENS;
    }
}

Eagle3RTInfo
extract_eagle_mode_from_config(ov::AnyMap& config, const std::filesystem::path& models_path) {
    Eagle3RTInfo eagle_rt_info;
    if (config.find("eagle3_mode") != config.end()) {
        eagle_rt_info.eagle3_mode = config.at("eagle3_mode").as<bool>();
        config.erase("eagle3_mode");
        if (config.find("hidden_layers_list") != config.end()) {
            eagle_rt_info.hidden_layers_list = config.at("hidden_layers_list").as<std::vector<int>>();
            config.erase("hidden_layers_list");
        } else {
            // compute the layers from number of hidden layers
            auto config_file_path = models_path / "config.json";
            if (!std::filesystem::exists(config_file_path))
                OPENVINO_THROW("cannot deduce layers for hidden layer extraction");
            std::ifstream file(config_file_path);

            nlohmann::json data = nlohmann::json::parse(file);
            using ov::genai::utils::read_json_param;
            int num_decoder_layers = 0;
            read_json_param(data, "num_hidden_layers", num_decoder_layers);
            OPENVINO_ASSERT(num_decoder_layers > 3, "num_decoder_layers is too small to deduce hidden layers for extraction");
            // The following default hidden layer selection corresponds to the EAGLE reference implementation:
            // https://github.com/SafeAILab/EAGLE/blob/0ea94696/eagle/model/modeling_llama_kv.py#L1138
            // These layers (2, num_decoder_layers / 2, num_decoder_layers - 3) are chosen to capture features from
            // early, middle, and late stages of the decoder, as recommended by the EAGLE authors.
            // If you wish to use different layers, provide the "hidden_layers_list" parameter in the config.
            eagle_rt_info.hidden_layers_list = { 2, num_decoder_layers / 2, num_decoder_layers - 3 };
        }
    }
    return eagle_rt_info;
}

}  // namespace speculative_decoding
}  // namespace genai
}  // namespace ov
