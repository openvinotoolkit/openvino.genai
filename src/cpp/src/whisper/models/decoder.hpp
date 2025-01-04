// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/runtime/core.hpp"

namespace ov::genai {
class WhisperDecoder {
public:
    static std::shared_ptr<WhisperDecoder> from_path(const std::filesystem::path& models_path,
                                                     const std::string& device,
                                                     const ov::AnyMap& properties);

    virtual std::pair<int64_t, float> detect_language(const ov::Tensor& encoder_hidden_state,
                                                      const int64_t decoder_start_token_id) = 0;

    virtual std::pair<ov::Tensor, float> decode(const ov::Tensor& encoder_hidden_state,
                                                const std::vector<int64_t>& input_ids,
                                                const size_t cache_position) = 0;

    virtual void reset_state() = 0;

    virtual ~WhisperDecoder();
};
}  // namespace ov::genai
