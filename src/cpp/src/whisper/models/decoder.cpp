// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include <filesystem>

#include "statefull_decoder.hpp"
#include "utils.hpp"
#include "with_past_decoder.hpp"

namespace ov::genai {
std::shared_ptr<WhisperDecoder> WhisperDecoder::from_path(const std::filesystem::path& models_path,
                                                          const std::string& device,
                                                          const ov::AnyMap& properties) {
    bool has_decoder_with_past = std::filesystem::exists(models_path / "openvino_decoder_with_past_model.xml");

    if (has_decoder_with_past) {
        return std::make_shared<WhisperWithPastDecoder>(models_path, device, properties);
    }

    return std::make_shared<WhisperStatefullDecoder>(models_path, device, properties);
}

/**
 * Encoder hidden states expected to be with batch 1
 * Copy encoder hidden state tensor from batch 1 to requested batch_size.
 * Set new encoder hidden states tensor to infer request.
 */
void WhisperDecoder::_set_encoder_hidden_states_tensor(const Tensor& encoder_hidden_state,
                                                       const size_t batch_size,
                                                       InferRequest& request) {
    const size_t current_batch_size = request.get_tensor("encoder_hidden_states").get_shape().at(0);
    // batch hasn't changed, skip
    if (current_batch_size == batch_size) {
        return;
    }

    OPENVINO_ASSERT(encoder_hidden_state.get_shape().at(0) == 1);
    Shape shape{encoder_hidden_state.get_shape()};
    shape[0] = batch_size;

    Tensor new_encoder_hidden_states{ov::element::f32, shape};

    auto new_encoder_hidden_states_data = new_encoder_hidden_states.data<float>();
    auto encoder_hidden_state_data = encoder_hidden_state.data<float>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * encoder_hidden_state.get_size();
        std::memcpy(new_encoder_hidden_states_data + batch_offset,
                    encoder_hidden_state_data,
                    encoder_hidden_state.get_byte_size());
    }

    request.set_tensor("encoder_hidden_states", new_encoder_hidden_states);
}

WhisperDecoder::~WhisperDecoder() = default;
}  // namespace ov::genai
