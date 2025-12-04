// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/runtime/tensor.hpp>
#include <vector>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "whisper.hpp"
#include "whisper/config.hpp"

namespace ov::genai {

std::pair<std::vector<std::string>, std::vector<std::vector<int64_t>>> split_tokens_on_spaces(
    const std::vector<int64_t>& tokens,
    ov::genai::Tokenizer& tokenizer);

std::vector<WhisperWordTiming> get_word_level_timestamps(const std::vector<Tensor>& encoder_attention_qks,
                                                         const WhisperConfig& model_config,
                                                         const size_t n_frames,
                                                         const std::vector<int64_t>& tokens,
                                                         ov::genai::Tokenizer& tokenizer,
                                                         const ov::genai::WhisperGenerationConfig& generation_config);
}  // namespace ov::genai