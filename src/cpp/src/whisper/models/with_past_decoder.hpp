// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "decoder.hpp"
#include "openvino/runtime/core.hpp"

namespace ov::genai {

class WhisperWithPastDecoder : public WhisperDecoder {
public:
    WhisperWithPastDecoder(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties);

    std::pair<int64_t, float> detect_language(const Tensor& encoder_hidden_state,
                                              const int64_t decoder_start_token_id) override;

    std::pair<Tensor, float> decode(const Tensor& encoder_hidden_state,
                                    const Tensor& input_ids,
                                    const Tensor& beam_idx) override;

    void reset_state() override;

private:
    ov::InferRequest m_request_decoder;
    ov::InferRequest m_request_decoder_with_past;
    size_t m_cache_position = 0;
    bool m_initial_past_key_value_set = false;
    bool m_past_key_value_linked = false;

    void _set_past_key_value(const Tensor& beam_idx);
};

}  // namespace ov::genai
