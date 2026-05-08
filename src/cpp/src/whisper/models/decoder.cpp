// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include <filesystem>
#include <limits>

#include "statefull_decoder.hpp"
#include "whisper/whisper_utils.hpp"

namespace ov::genai {
std::shared_ptr<WhisperDecoder> WhisperDecoder::from_path(const std::filesystem::path& models_path,
                                                          const std::string& device,
                                                          const ov::AnyMap& properties,
                                                          const ov::PartialShape& lhs_shape,
                                                          const bool decompose_cross_attention_spda_ops) {
    return std::make_shared<WhisperStatefullDecoder>(models_path,
                                                     device,
                                                     properties,
                                                     lhs_shape,
                                                     decompose_cross_attention_spda_ops);
}

std::pair<int64_t, float> WhisperDecoder::detect_language(const ov::Tensor& encoder_hidden_state,
                                                          const WhisperGenerationConfig& config) {
    Tensor input_ids_tensor = create_host_tensor(ov::element::i64, {1, 1});
    input_ids_tensor.data<int64_t>()[0] = config.decoder_start_token_id;

    Tensor beam_idx_tensor = create_host_tensor(ov::element::i32, {1});
    beam_idx_tensor.data<int32_t>()[0] = 0;

    const auto infer_start = std::chrono::steady_clock::now();
    start_async(encoder_hidden_state, input_ids_tensor, beam_idx_tensor);

    auto output_tensor = wait();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    auto logits_data = output_tensor.data<float>();

    int64_t output_token = -1;
    float max_prob = -std::numeric_limits<float>::infinity();

    for (auto [_, lang_token] : config.lang_to_id) {
        auto prob = logits_data[lang_token];
        if (prob > max_prob) {
            max_prob = prob;
            output_token = lang_token;
        }
    }

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
