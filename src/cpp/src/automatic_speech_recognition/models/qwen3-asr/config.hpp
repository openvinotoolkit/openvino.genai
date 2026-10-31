// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

namespace ov {
namespace genai {

/**
 * @brief Structure to keep Qwen3-ASR config parameters.
 */
class Qwen3ASRConfig {
public:
    explicit Qwen3ASRConfig(const std::filesystem::path& json_path);

    size_t n_window = 100;
};

}  // namespace genai
}  // namespace ov
