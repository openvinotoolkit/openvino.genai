// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <set>

namespace ov::genai {

/// @brief Repetition penalty over the last generated tokens only: the prompt and the ignored tokens never count.
struct RepetitionWindow {
    size_t size = 0;
    std::set<int64_t> ignored_token_ids;
};

/// @brief Sampling rules of one request that GenerationConfig cannot express, set by the model code that adds it.
struct SamplingOverrides {
    /// @brief Replaces the suppressed token ids of the pipeline for this request.
    std::optional<std::set<int64_t>> suppressed_token_ids;
    /// @brief Replaces the repetition penalty history: prompt and all generated tokens.
    std::optional<RepetitionWindow> repetition_window;
};

}  // namespace ov::genai
