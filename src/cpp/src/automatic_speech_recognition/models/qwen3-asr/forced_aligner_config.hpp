// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ov {
namespace genai {

/**
 * @brief Forced-aligner-specific parameters read from the Qwen3 forced-aligner config.json.
 *
 * Only the fields the forced aligner needs are parsed here. Encoder geometry (n_window) is
 * read independently by Qwen3ASREncoder / Qwen3ASRConfig from the same config.json.
 * support_languages is exposed verbatim; language canonicalization is the aligner's responsibility.
 */
class Qwen3ForcedAlignerConfig {
public:
    explicit Qwen3ForcedAlignerConfig(const std::filesystem::path& json_path);

    int64_t timestamp_token_id;
    float timestamp_segment_ms;
    size_t classify_num;
    std::vector<std::string> support_languages;
};

}  // namespace genai
}  // namespace ov
