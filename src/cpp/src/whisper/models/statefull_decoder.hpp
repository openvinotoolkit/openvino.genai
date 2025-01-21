// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "decoder.hpp"
#include "openvino/runtime/core.hpp"

namespace ov::genai {

class WhisperStatefullDecoder : public WhisperDecoder {
public:
    WhisperStatefullDecoder(const std::filesystem::path& models_path,
                            const std::string& device,
                            const ov::AnyMap& properties);

    std::pair<int64_t, float> detect_language(const Tensor& encoder_hidden_state,
                                              const int64_t decoder_start_token_id) override;

    std::pair<Tensor, float> decode(const Tensor& encoder_hidden_state,
                                    const Tensor& input_ids,
                                    const Tensor& beam_idx) override;

    void reset_state() override;

private:
    void _set_encoder_hidden_states_tensor(const Tensor& encoder_hidden_state, const size_t batch_size);
    void _reset_encoder_past_key_values_states(const Tensor& encoder_hidden_state, const size_t batch_size);
    void _set_cache_position_tensor(const size_t seq_len);

private:
    ov::InferRequest m_request;
};
}  // namespace ov::genai
