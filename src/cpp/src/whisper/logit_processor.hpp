// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "openvino/genai/whisper_generation_config.hpp"

namespace ov {
namespace genai {

void do_suppress_tokens(ov::Tensor& logits, const size_t batch_idx, const std::vector<int64_t>& suppress_tokens);

void process_whisper_timestamp_logits(ov::Tensor& logits,
                                      const size_t batch_idx,
                                      const ov::genai::WhisperGenerationConfig& config,
                                      const std::vector<int64_t>& generated_tokens,
                                      bool initial_step = false);

}  // namespace genai
}  // namespace ov
