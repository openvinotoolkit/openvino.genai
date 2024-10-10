// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ov {
namespace genai {

/**
 * @brief Structure to keep whisper config parameters.
 */
class WhisperConfig {
public:
    explicit WhisperConfig(const std::string& json_path);

    size_t max_source_positions = 1500;
};

}  // namespace genai
}  // namespace ov
