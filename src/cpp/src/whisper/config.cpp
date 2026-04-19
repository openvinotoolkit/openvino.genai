// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper/config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace genai {

WhisperConfig::WhisperConfig(const std::filesystem::path& json_path) {
    // config not found. Skip parameters initialization from file, use defaults.
    if (!std::filesystem::exists(json_path)) {
        return;
    }

    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path, "' with config");

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "max_source_positions", max_source_positions);
    read_json_param(data, "model_type", model_type);

    // Parse Qwen3-ASR specific config from thinker_config section
    if (model_type == "qwen3_asr" && data.contains("thinker_config")) {
        const auto& thinker = data["thinker_config"];
        read_json_param(thinker, "audio_token_id", audio_token_id);
        read_json_param(thinker, "audio_start_token_id", audio_start_token_id);
        read_json_param(thinker, "audio_end_token_id", audio_end_token_id);

        // Encoder max_source_positions is in audio_config
        if (thinker.contains("audio_config")) {
            read_json_param(thinker["audio_config"], "max_source_positions", max_source_positions);
        }
    }
}

}  // namespace genai
}  // namespace ov
