// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include "nlohmann/json.hpp"
#include "qwen3_5config.hpp"
#include "openvino/core/except.hpp"
#include "json_utils.hpp"

namespace ov::genai::module {

Qwen3_5VisionConfig Qwen3_5VisionConfig::from_json_file(const std::filesystem::path &path) {
    std::ifstream json_file(path);
    if (!json_file.is_open()) {
        OPENVINO_THROW("Failed to open vision config file: ", path.string());
    }
    nlohmann::json data;
    json_file >> data;
    Qwen3_5VisionConfig cfg;
    using ov::genai::utils::read_json_param;
    read_json_param(data, "vision_config.model_type", cfg.model_type);
    read_json_param(data, "vision_config.depth", cfg.depth);
    read_json_param(data, "vision_config.hidden_size", cfg.hidden_size);
    read_json_param(data, "vision_config.hidden_act", cfg.hidden_act);
    read_json_param(data, "vision_config.intermediate_size", cfg.intermediate_size);
    read_json_param(data, "vision_config.num_heads", cfg.num_heads);
    read_json_param(data, "vision_config.in_channels", cfg.in_channels);
    read_json_param(data, "vision_config.patch_size", cfg.patch_size);
    read_json_param(data, "vision_config.spatial_merge_size", cfg.spatial_merge_size);
    read_json_param(data, "vision_config.temporal_patch_size", cfg.temporal_patch_size);
    read_json_param(data, "vision_config.out_hidden_size", cfg.out_hidden_size);
    read_json_param(data, "vision_config.num_position_embeddings", cfg.num_position_embeddings);
    read_json_param(data, "vision_config.deepstack_visual_indexes", cfg.deepstack_visual_indexes);
    read_json_param(data, "vision_config.initializer_range", cfg.initializer_range);

    return cfg;
}

int32_t Qwen3_5VisionConfig::head_dim() const {
    if (num_heads <= 0) {
        return 0;
    }
    return hidden_size / num_heads;
}

Qwen3_5VisionPreprocessConfig Qwen3_5VisionPreprocessConfig::from_json_file(const std::filesystem::path &path) {
    std::ifstream json_file(path);
    if (!json_file.is_open()) {
        OPENVINO_THROW("Failed to open vision preprocess config file: ", path.string());
    }
    nlohmann::json data;
    json_file >> data;
    Qwen3_5VisionPreprocessConfig cfg;
    using ov::genai::utils::read_json_param;
    read_json_param(data, "size.shortest_edge", cfg.min_pixels);
    read_json_param(data, "size.longest_edge", cfg.max_pixels);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "temporal_patch_size", cfg.temporal_patch_size);
    read_json_param(data, "merge_size", cfg.merge_size);
    read_json_param(data, "image_mean", cfg.image_mean);
    read_json_param(data, "image_std", cfg.image_std);

    return cfg;
}

}