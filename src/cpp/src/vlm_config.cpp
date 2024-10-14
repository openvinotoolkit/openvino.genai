// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/vlm_config.hpp"
#include "utils.hpp"
#include <fstream>

ov::genai::VLMConfig::VLMConfig(const std::filesystem::path& json_path) {
    std::ifstream stream(json_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '" + json_path.string() + "' with processor config");
    nlohmann::json parsed = nlohmann::json::parse(stream);
    using ov::genai::utils::read_json_param;
    model_type = to_vlm_model_type(parsed.at("model_type"));
    read_json_param(parsed, "hidden_size", hidden_size);
    read_json_param(parsed, "scale_emb", scale_emb);
    read_json_param(parsed, "query_num", query_num);
    read_json_param(parsed, "use_image_id", use_image_id);
}
