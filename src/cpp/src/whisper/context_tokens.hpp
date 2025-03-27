// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/whisper_generation_config.hpp"

namespace ov {
namespace genai {

struct WhisperContextTokens {
    std::vector<int64_t> initial_prompt;
    std::vector<int64_t> hotwords;
};

std::pair<WhisperContextTokens, float> prepare_context_tokens(const WhisperGenerationConfig& config,
                                                              Tokenizer& tokenizer);

std::vector<int64_t> get_prompt_tokens(const WhisperContextTokens& context_tokens,
                                       const WhisperGenerationConfig& config,
                                       size_t chunk_offset);

}  // namespace genai
}  // namespace ov
