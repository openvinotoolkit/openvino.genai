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

    std::pair<int64_t, float> detect_language(const ov::Tensor& encoder_hidden_state,
                                              const int64_t decoder_start_token_id) override;

    std::pair<ov::Tensor, float> decode(const ov::Tensor& encoder_hidden_state,
                                        const std::vector<int64_t>& input_ids,
                                        const size_t cache_position) override;

    void reset_state() override;

private:
    ov::InferRequest m_request_decoder;
    ov::InferRequest m_request_decoder_with_past;
    bool m_decoder_with_past_kv_value_set = false;
};

}  // namespace ov::genai
