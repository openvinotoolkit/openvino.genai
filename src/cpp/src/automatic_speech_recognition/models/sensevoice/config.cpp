// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace genai {

SenseVoiceSmallConfig::SenseVoiceSmallConfig(const std::filesystem::path& json_path) {
    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open SenseVoiceSmall config.json in '", json_path.string(), "'");
    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "blank_id", blank_id);
    read_json_param(data, "lid_dict", lid_dict);
    read_json_param(data, "textnorm_dict", textnorm_dict);
    read_json_param(data, "dither", dither);

    validate();
}

void SenseVoiceSmallConfig::validate() const {
    OPENVINO_ASSERT(blank_id >= 0, "SenseVoiceSmall config.json must define a non-negative 'blank_id'");
    OPENVINO_ASSERT(!lid_dict.empty() && lid_dict.count("auto") != 0,
                    "SenseVoiceSmall config.json 'lid_dict' must be non-empty and contain the 'auto' language id");
    OPENVINO_ASSERT(textnorm_dict.count("woitn") != 0 && textnorm_dict.count("withitn") != 0,
                    "SenseVoiceSmall config.json 'textnorm_dict' must contain both the 'woitn' and 'withitn' ids");
}

}  // namespace genai
}  // namespace ov
