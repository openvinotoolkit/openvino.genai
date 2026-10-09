// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

namespace ov {
namespace genai {

/**
 * @brief Structure to keep SenseVoiceSmall config parameters.
 */
class SenseVoiceSmallConfig {
public:
    explicit SenseVoiceSmallConfig(const std::filesystem::path& json_path);

    void validate() const;

    int64_t blank_id = -1;
    std::map<std::string, int64_t> lid_dict;
    std::map<std::string, int64_t> textnorm_dict;
    float dither = 1.0f;
};

}  // namespace genai
}  // namespace ov
