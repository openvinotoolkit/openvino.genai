// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

namespace ov {
namespace genai {

/**
 * @brief Structure to keep whisper config parameters.
 */
class WhisperConfig {
public:
    explicit WhisperConfig(const std::filesystem::path& json_path);

    size_t max_source_positions = 1500;
};

}  // namespace genai
}  // namespace ov
