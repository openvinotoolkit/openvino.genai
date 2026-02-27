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

std::vector<ov::genai::WhisperWordTiming> add_word_level_timestamps(const std::vector<int64_t>& sot_tokens,
                                                                    const std::vector<int64_t>& text_tokens,
                                                                    ov::genai::Tokenizer& tokenizer,
                                                                    std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                                                    const ov::Tensor& hidden_state_tensor,
                                                                    const ov::genai::WhisperGenerationConfig& config,
                                                                    const size_t n_frames,
                                                                    const float chunk_time_offset);

std::vector<ov::genai::WhisperWordTiming> add_word_level_timestamps(const std::vector<int64_t>& sot_tokens,
                                                                    const std::vector<int64_t>& input_tokens,
                                                                    ov::genai::Tokenizer& tokenizer,
                                                                    ov::InferRequest& decoder,
                                                                    const ov::Tensor& hidden_state_tensor,
                                                                    const ov::genai::WhisperGenerationConfig& config,
                                                                    const size_t n_active_frames,
                                                                    const float chunk_time_offset);

void decompose_scaled_dot_product_attention_for_whisper(std::shared_ptr<ov::Model> model);

void add_cross_attention_qk_scaled_scores_outputs_for_whisper(std::shared_ptr<ov::Model> model);
}  // namespace ov::genai
