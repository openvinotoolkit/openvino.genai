// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper/config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "openvino/core/except.hpp"

#include "json_utils.hpp"

namespace ov {
namespace genai {

WhisperConfig::WhisperConfig(const std::filesystem::path& json_path) {
    // preprocessor_config.json not found. Skip parameters initialization from file, use defaults.
    if (!std::filesystem::exists(json_path)) {
        return;
    }

    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path, "' with config");

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "max_source_positions", max_source_positions);
}

}  // namespace genai
}  // namespace ov
