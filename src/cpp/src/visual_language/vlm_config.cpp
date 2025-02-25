// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vlm_config.hpp"
#include "json_utils.hpp"

#include <fstream>

ov::genai::VLMConfig::VLMConfig(const std::filesystem::path& json_path) {
    std::ifstream stream(json_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", json_path, "' with processor config");
    nlohmann::json parsed = nlohmann::json::parse(stream);
    using ov::genai::utils::read_json_param;
    model_type = to_vlm_model_type(parsed.at("model_type"));
    read_json_param(parsed, "hidden_size", hidden_size);
    read_json_param(parsed, "scale_emb", scale_emb);
    read_json_param(parsed, "query_num", query_num);
    read_json_param(parsed, "use_image_id", use_image_id);

    // Setting llava_next specific config params
    read_json_param(parsed, "image_newline", image_newline);
    if (parsed.contains("vision_config")) {
        read_json_param(parsed.at("vision_config"), "patch_size", vision_config_patch_size);
    }
    // phi3_v
    if (parsed.contains("sub_GN")) {
        sub_GN = parsed.at("sub_GN").get<std::vector<std::vector<std::vector<std::vector<float>>>>>().at(0).at(0).at(0);
    }
    OPENVINO_ASSERT(sub_GN.size() == 4096);
    if (parsed.contains("glb_GN")) {
        glb_GN = parsed.at("glb_GN").get<std::vector<std::vector<std::vector<float>>>>().at(0).at(0);
    }
    OPENVINO_ASSERT(glb_GN.size() == 4096);
}
