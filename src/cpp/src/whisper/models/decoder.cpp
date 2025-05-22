// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include <filesystem>

#include "statefull_decoder.hpp"
#include "whisper/whisper_utils.hpp"
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

std::pair<int64_t, float> WhisperDecoder::detect_language(const ov::Tensor& encoder_hidden_state,
                                                          const int64_t decoder_start_token_id) {
    Tensor input_ids_tensor = create_host_tensor(ov::element::i64, {1, 1});
    input_ids_tensor.data<int64_t>()[0] = decoder_start_token_id;

    Tensor beam_idx_tensor = create_host_tensor(ov::element::i32, {1});
    beam_idx_tensor.data<int32_t>()[0] = 0;

    const auto infer_start = std::chrono::steady_clock::now();
    start_async(encoder_hidden_state, input_ids_tensor, beam_idx_tensor);

    auto output_tensor = wait();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    reset_state();

    return {output_token, infer_ms};
}

/**
 * Encoder hidden states expected to be with batch 1
 * Expand encoder hidden state tensor from batch 1 to requested batch_size.
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

    if (batch_size == 1) {
        request.set_tensor("encoder_hidden_states", encoder_hidden_state);
        return;
    }

    Shape shape{encoder_hidden_state.get_shape()};
    shape[0] = batch_size;

    Tensor new_encoder_hidden_states = create_host_tensor(ov::element::f32, shape);

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

ov::Tensor WhisperDecoder::create_host_tensor(const element::Type element_type, const Shape& shape) {
    return ov::Tensor(element_type, shape);
}

WhisperDecoder::~WhisperDecoder() = default;
}  // namespace ov::genai
