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
    bool m_initial_step = true;
    bool m_decoder_with_past_kv_value_set = false;
    size_t m_cache_position = 0;

    void _set_encoder_hidden_states_tensor(const Tensor& encoder_hidden_state,
                                           const size_t batch_size,
                                           InferRequest& request);
};

}  // namespace ov::genai
