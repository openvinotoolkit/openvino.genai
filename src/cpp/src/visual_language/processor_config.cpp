// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "processor_config.hpp"
#include "json_utils.hpp"

#include <fstream>

ov::genai::ProcessorConfig::ProcessorConfig(const std::filesystem::path& json_path) {
    std::ifstream stream(json_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", json_path, "' with processor config");
    nlohmann::json parsed = nlohmann::json::parse(stream);
    using ov::genai::utils::read_json_param;
    read_json_param(parsed, "patch_size", patch_size); // For llava - stored in config.json vision_config
    read_json_param(parsed, "scale_resolution", scale_resolution);
    read_json_param(parsed, "max_slice_nums", max_slice_nums);
    if (parsed.contains("norm_mean")) {
        norm_mean = parsed.at("norm_mean").get<std::array<float, 3>>();
    }
    if (parsed.contains("norm_std")) {
        norm_std = parsed.at("norm_std").get<std::array<float, 3>>();
    }
    
    // Setting llava config params
    if (parsed.contains("image_mean")) {
        image_mean = parsed.at("image_mean").get<std::array<float, 3>>();
    }
    if (parsed.contains("image_std")) {
        image_std = parsed.at("image_std").get<std::array<float, 3>>();
    }

    if (parsed.contains("crop_size")) {
        crop_size_height = parsed.at("crop_size").at("height");
        crop_size_width = parsed.at("crop_size").at("width");
    }
    if (parsed.contains("size") && parsed.at("size").contains("shortest_edge")) {
        size_shortest_edge = parsed.at("size").at("shortest_edge");
    }

    // Setting llava-next config params
    if (parsed.contains("image_grid_pinpoints")) {
        image_grid_pinpoints = parsed.at("image_grid_pinpoints").get<std::vector<std::pair<int, int>>>();
    }
    read_json_param(parsed, "num_crops", phi3_v.num_crops);
    if (parsed.contains("img_processor")) {
        phi3_v.num_img_tokens = parsed.at("img_processor").at("num_img_tokens");
    }

    // Setting qwen2vl config params
    read_json_param(parsed, "min_pixels", min_pixels);
    read_json_param(parsed, "max_pixels", max_pixels);
    read_json_param(parsed, "temporal_patch_size", temporal_patch_size);
    read_json_param(parsed, "merge_size", merge_size);

    // Setting gemma3-4b-it config params
    if (parsed.contains("size") && parsed.at("size").contains("height")) {
        size_height = parsed.at("size").at("height");
    }
    if (parsed.contains("size") && parsed.at("size").contains("width")) {
        size_width = parsed.at("size").at("width");
    }
}
