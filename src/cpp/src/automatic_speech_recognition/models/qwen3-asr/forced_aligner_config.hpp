// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ov {
namespace genai {

/**
 * @brief Configuration parameters required by the Qwen3 forced aligner.
 *
 * Reads forced-aligner-specific fields from config.json. Encoder parameters are handled
 * separately by Qwen3ASRConfig.
 */
class Qwen3ForcedAlignerConfig {
public:
    explicit Qwen3ForcedAlignerConfig(const std::filesystem::path& json_path);

    void validate() const;

    int64_t timestamp_token_id;
    float timestamp_segment_ms;
    size_t classify_num;
    std::vector<std::string> support_languages;
};

}  // namespace genai
}  // namespace ov
