// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vlm_config.hpp"
#include "json_utils.hpp"

#include <fstream>

namespace ov::genai {

namespace {

VLMModelType to_vlm_model_type(const std::string& value) {
    static const std::unordered_map<std::string, VLMModelType> model_types_map = {
        {"minicpmv", VLMModelType::MINICPM},
        {"llava", VLMModelType::LLAVA},
        {"llava_next", VLMModelType::LLAVA_NEXT},
        {"internvl_chat", VLMModelType::INTERNVL_CHAT},
        {"phi3_v", VLMModelType::PHI3_V},
        {"phi4mm", VLMModelType::PHI4MM},
        {"qwen2_vl", VLMModelType::QWEN2_VL},
        {"qwen2_5_vl", VLMModelType::QWEN2_5_VL},
    };

    auto it = model_types_map.find(value);
    if (it != model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' VLM model type");
}

} // namespace

VLMConfig::VLMConfig(const std::filesystem::path& json_path) {
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
    // phi3_v and phi4mm
    if (parsed.contains("sub_GN")) {
        sub_GN = parsed.at("sub_GN").get<std::vector<std::vector<std::vector<std::vector<float>>>>>().at(0).at(0).at(0);
    }
    if (model_type == VLMModelType::PHI3_V) {
        OPENVINO_ASSERT(sub_GN.size() == 4096);
    } else if (model_type == VLMModelType::PHI4MM) {
        OPENVINO_ASSERT(sub_GN.size() == 1152);
    }
    if (parsed.contains("glb_GN")) {
        glb_GN = parsed.at("glb_GN").get<std::vector<std::vector<std::vector<float>>>>().at(0).at(0);
    }
    if (model_type == VLMModelType::PHI3_V) {
        OPENVINO_ASSERT(glb_GN.size() == 4096);
    } else if (model_type == VLMModelType::PHI4MM) {
        OPENVINO_ASSERT(glb_GN.size() == 1152);
    }
    // Qwen2.5VL
    if (parsed.contains("vision_config")) {
        read_json_param(parsed.at("vision_config"), "window_size", vision_config_window_size);
    }
}

} // namespace ov::genai
