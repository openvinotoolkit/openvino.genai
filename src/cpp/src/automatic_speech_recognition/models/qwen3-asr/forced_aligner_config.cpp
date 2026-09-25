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

    OPENVINO_ASSERT(config.contains("timestamp_token_id") && config.contains("timestamp_segment_time"),
                    "Model at '", json_path.string(),
                    "' is not a Qwen3 forced aligner: config.json lacks timestamp_token_id / "
                    "timestamp_segment_time.");

    OPENVINO_ASSERT(config.contains("thinker_config") && config.at("thinker_config").contains("classify_num"),
                    "Model at '", json_path.string(),
                    "' is not a Qwen3 forced aligner: config.json lacks thinker_config.classify_num.");

    const nlohmann::json& timestamp_token_id_json = config.at("timestamp_token_id");
    OPENVINO_ASSERT(timestamp_token_id_json.is_number_integer(),
                    "Forced-aligner timestamp_token_id must be a non-negative integer. Got: ",
                    timestamp_token_id_json.dump(),
                    ".");
    read_json_param(config, "timestamp_token_id", timestamp_token_id);

    const nlohmann::json& segment_time_json = config.at("timestamp_segment_time");
    OPENVINO_ASSERT(segment_time_json.is_number(),
                    "Forced-aligner timestamp_segment_time must be a positive number. Got: ",
                    segment_time_json.dump(),
                    ".");
    read_json_param(config, "timestamp_segment_time", timestamp_segment_ms);

    const nlohmann::json& classify_num_json = config.at("thinker_config").at("classify_num");
    OPENVINO_ASSERT(classify_num_json.is_number_integer(),
                    "Forced-aligner thinker_config.classify_num must be a positive integer. Got: ",
                    classify_num_json.dump(), ".");
    int64_t classify_num_value = 0;
    read_json_param(config, "thinker_config.classify_num", classify_num_value);
    OPENVINO_ASSERT(classify_num_value > 0,
                    "Forced-aligner thinker_config.classify_num must be a positive integer. Got: ",
                    classify_num_json.dump(), ".");
    classify_num = static_cast<size_t>(classify_num_value);

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
