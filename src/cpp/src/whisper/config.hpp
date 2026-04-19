// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ov {
namespace genai {

/**
 * @brief Structure to keep whisper config parameters.
 */
class WhisperConfig {
public:
    explicit WhisperConfig(const std::filesystem::path& json_path);

    // Common parameters
    std::string model_type = "whisper";
    size_t max_source_positions = 1500;

    // Qwen3-ASR specific parameters (populated only when model_type == "qwen3_asr")
    int64_t audio_token_id = -1;
    int64_t audio_start_token_id = -1;
    int64_t audio_end_token_id = -1;

    // Helper to check if this is a Qwen3-ASR model
    bool is_qwen3_asr() const {
        return model_type == "qwen3_asr";
    }
};

}  // namespace genai
}  // namespace ov
