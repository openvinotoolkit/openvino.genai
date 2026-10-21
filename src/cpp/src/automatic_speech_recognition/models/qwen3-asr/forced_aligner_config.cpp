// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "forced_aligner_config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace genai {

Qwen3ForcedAlignerConfig::Qwen3ForcedAlignerConfig(const std::filesystem::path& json_path) {
    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path, "'");
    nlohmann::json config = nlohmann::json::parse(f);

    read_json_param(config, "timestamp_token_id", timestamp_token_id);
    read_json_param(config, "timestamp_segment_time", timestamp_segment_ms);
    read_json_param(config, "thinker_config.classify_num", classify_num);
    read_json_param(config, "support_languages", support_languages);

    validate();
}

void Qwen3ForcedAlignerConfig::validate() const {
    OPENVINO_ASSERT(timestamp_token_id >= 0,
                    "Forced-aligner timestamp_token_id must be a non-negative integer. Got: ",
                    timestamp_token_id,
                    ".");

    OPENVINO_ASSERT(timestamp_segment_ms > 0.0f,
                    "Forced-aligner timestamp_segment_time must be a positive number. Got: ",
                    timestamp_segment_ms,
                    ".");

    OPENVINO_ASSERT(classify_num > 0,
                    "Forced-aligner thinker_config.classify_num must be a positive integer. Got: ",
                    classify_num,
                    ".");
}

}  // namespace genai
}  // namespace ov
