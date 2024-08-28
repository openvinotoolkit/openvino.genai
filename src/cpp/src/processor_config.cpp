// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/processor_config.hpp"
#include "utils.hpp"
#include <fstream>

ov::genai::ProcessorConfig::ProcessorConfig(const std::filesystem::path& json_path) {
    std::ifstream stream(json_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '" + json_path.string() + "' with processor config");
    nlohmann::json parsed = nlohmann::json::parse(stream);
    using ov::genai::utils::read_json_param;
    read_json_param(parsed, "patch_size", patch_size);
    read_json_param(parsed, "scale_resolution", scale_resolution);
    read_json_param(parsed, "max_slice_nums", max_slice_nums);
    if (parsed.contains("norm_mean")) {
        norm_mean = parsed.at("norm_mean").get<std::array<float, 3>>();
    }
    if (parsed.contains("norm_std")) {
        norm_std = parsed.at("norm_std").get<std::array<float, 3>>();
    }
}
