// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "forced_aligner_config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "openvino/core/except.hpp"

namespace ov {
namespace genai {

Qwen3ForcedAlignerConfig::Qwen3ForcedAlignerConfig(const std::filesystem::path& json_path) {
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
    const nlohmann::json& classify_num_json = config.at("thinker_config").at("classify_num");
    OPENVINO_ASSERT(classify_num_json.is_number_integer() && classify_num_json.get<int64_t>() > 0,
                    "Forced-aligner thinker_config.classify_num must be a positive integer. Got: ",
                    classify_num_json.dump(), ".");
    classify_num = static_cast<size_t>(classify_num_json.get<int64_t>());

    const nlohmann::json& timestamp_token_id_json = config.at("timestamp_token_id");
    OPENVINO_ASSERT(timestamp_token_id_json.is_number_integer(),
                    "Forced-aligner timestamp_token_id must be an integer. Got: ",
                    timestamp_token_id_json.dump(),
                    ".");
    timestamp_token_id = timestamp_token_id_json.get<int64_t>();

    const nlohmann::json& segment_time_json = config.at("timestamp_segment_time");
    OPENVINO_ASSERT(segment_time_json.is_number() && segment_time_json.get<double>() > 0.0,
                    "Forced-aligner timestamp_segment_time must be a positive number. Got: ",
                    segment_time_json.dump(),
                    ".");
    timestamp_segment_ms = segment_time_json.get<float>();

    if (config.contains("support_languages") && config.at("support_languages").is_array()) {
        for (const auto& language : config.at("support_languages")) {
            support_languages.push_back(language.get<std::string>());
        }
    }
}

}  // namespace genai
}  // namespace ov
