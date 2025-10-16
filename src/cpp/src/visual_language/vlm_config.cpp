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
        {"minicpmo", VLMModelType::MINICPM},
        {"llava", VLMModelType::LLAVA},
        {"llava-qwen2", VLMModelType::NANOLLAVA},
        {"llava_next", VLMModelType::LLAVA_NEXT},
        {"llava_next_video", VLMModelType::LLAVA_NEXT_VIDEO},
        {"internvl_chat", VLMModelType::INTERNVL_CHAT},
        {"phi3_v", VLMModelType::PHI3_V},
        {"phi4mm", VLMModelType::PHI4MM},
        {"qwen2_vl", VLMModelType::QWEN2_VL},
        {"qwen2_5_vl", VLMModelType::QWEN2_5_VL},
        {"gemma3", VLMModelType::GEMMA3},
    };

    auto it = model_types_map.find(value);
    if (it != model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' VLM model type");
}

void assert_size(size_t size, VLMModelType model_type) {
    if (model_type == VLMModelType::PHI3_V) {
        OPENVINO_ASSERT(size == 4096, "Expected size 4096 for PHI3_V model type");
    }
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
    read_json_param(parsed, "vision_config.patch_size", vision_config_patch_size);

    // phi3_v and phi4mm
    if (parsed.contains("sub_GN") && parsed.at("sub_GN").is_array()) {
        sub_GN = parsed.at("sub_GN").get<std::vector<std::vector<std::vector<std::vector<float>>>>>().at(0).at(0).at(0);
    }
    assert_size(sub_GN.size(), model_type);
    if (parsed.contains("glb_GN") && parsed.at("glb_GN").is_array()) {
        glb_GN = parsed.at("glb_GN").get<std::vector<std::vector<std::vector<float>>>>().at(0).at(0);
    }
    assert_size(glb_GN.size(), model_type);

    // Qwen2.5VL
    read_json_param(parsed, "vision_config.window_size", vision_config_window_size);
}

} // namespace ov::genai
