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

    read_json_param(parsed, "norm_mean", norm_mean);
    read_json_param(parsed, "norm_std", norm_std);
    
    // Setting llava config params
    read_json_param(parsed, "image_mean", image_mean);
    read_json_param(parsed, "image_std", image_std);
    read_json_param(parsed, "crop_size.height", crop_size_height);
    read_json_param(parsed, "crop_size.width", crop_size_width);
    read_json_param(parsed, "size.shortest_edge", size_shortest_edge);

    // Setting llava-next config params
    read_json_param(parsed, "image_grid_pinpoints", image_grid_pinpoints);
    read_json_param(parsed, "num_crops", phi3_v.num_crops);

    // Setting phi3_v config params
    read_json_param(parsed, "img_processor.num_img_tokens", phi3_v.num_img_tokens);

    // Setting qwen2vl config params
    read_json_param(parsed, "min_pixels", min_pixels);
    read_json_param(parsed, "max_pixels", max_pixels);
    read_json_param(parsed, "temporal_patch_size", temporal_patch_size);
    read_json_param(parsed, "merge_size", merge_size);

    // Setting gemma3-4b-it config params
    read_json_param(parsed, "size.height", size_height);
    read_json_param(parsed, "size.width", size_width);
}
