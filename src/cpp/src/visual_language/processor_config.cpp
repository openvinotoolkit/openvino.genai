// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "processor_config.hpp"
#include "json_utils.hpp"
#include "utils.hpp"

ov::genai::ProcessorConfig::ProcessorConfig(const nlohmann::json& parsed) {
    using ov::genai::utils::read_json_param;
    read_json_param(parsed, "image_size", image_size);
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
    
    // Setting qwen3_vl config params
    // qwen3_vl uses size.shortest_edge and size.longest_edge instead of min_pixels and max_pixels
    if (!parsed.contains("min_pixels") && !parsed.contains("max_pixels") ||
        parsed["min_pixels"].is_null() && parsed["max_pixels"].is_null()
    ) {
        read_json_param(parsed, "size.shortest_edge", min_pixels);
        read_json_param(parsed, "size.longest_edge", max_pixels);
    }

    // Setting gemma3-4b-it config params
    read_json_param(parsed, "size.height", size_height);
    read_json_param(parsed, "size.width", size_width);
}

ov::genai::ProcessorConfig::ProcessorConfig(const std::filesystem::path& json_path)
    : ProcessorConfig([&json_path] {
        std::ifstream stream(json_path);
        OPENVINO_ASSERT(stream.is_open(), "Failed to open '", json_path, "' with processor config");
        return nlohmann::json::parse(stream);
    }()) {}

ov::genai::ProcessorConfig ov::genai::ProcessorConfig::from_any_map(
    const ov::AnyMap& config_map,
    const ProcessorConfig& initial
) {
    auto iter = config_map.find("processor_config");
    ProcessorConfig extracted_config = config_map.end() != iter ?
        iter->second.as<ProcessorConfig>() : initial;
    using ov::genai::utils::read_anymap_param;
    read_anymap_param(config_map, "patch_size", extracted_config.patch_size);
    read_anymap_param(config_map, "scale_resolution", extracted_config.scale_resolution);
    read_anymap_param(config_map, "max_slice_nums", extracted_config.max_slice_nums);
    read_anymap_param(config_map, "norm_mean", extracted_config.norm_mean);
    read_anymap_param(config_map, "norm_std", extracted_config.norm_std);
    return extracted_config;
}
