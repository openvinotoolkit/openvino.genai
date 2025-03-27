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

    std::pair<int64_t, float> detect_language(const Tensor& encoder_hidden_state, const int64_t decoder_start_token_id);

    virtual void start_async(const Tensor& encoder_hidden_state, const Tensor& input_ids, const Tensor& beam_idx) = 0;
    
    virtual Tensor wait() = 0;

    virtual void reset_state() = 0;

    virtual ~WhisperDecoder();

    virtual ov::Tensor create_host_tensor(const element::Type element_type, const Shape& shape);

protected:
    void _set_encoder_hidden_states_tensor(const Tensor& encoder_hidden_state,
                                           const size_t batch_size,
                                           InferRequest& request);
};
}  // namespace ov::genai
