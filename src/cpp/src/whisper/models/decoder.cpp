// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <numeric>

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

std::pair<std::vector<int64_t>, float> WhisperDecoder::detect_languages(const ov::Tensor& encoder_hidden_state,
                                                                        const WhisperGenerationConfig& config) {
    // Language detection uses one decoder row per audio; beam expansion occurs only during generation.
    const size_t batch_size = encoder_hidden_state.get_shape().at(0);
    OPENVINO_ASSERT(batch_size > 0, "Language detection requires at least one encoder hidden state row.");

    Tensor input_ids_tensor = create_host_tensor(ov::element::i64, {batch_size, 1});
    std::fill_n(input_ids_tensor.data<int64_t>(), batch_size, config.decoder_start_token_id);

    Tensor beam_idx_tensor = create_host_tensor(ov::element::i32, {batch_size});
    std::iota(beam_idx_tensor.data<int32_t>(), beam_idx_tensor.data<int32_t>() + batch_size, 0);

    const auto infer_start = std::chrono::steady_clock::now();
    start_async(encoder_hidden_state, input_ids_tensor, beam_idx_tensor);

    auto output_tensor = wait();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    const auto logits_shape = output_tensor.get_shape();
    const size_t seq_len = logits_shape.at(1);
    const size_t vocab_size = logits_shape.back();
    const auto* logits_data = output_tensor.data<float>();

    // Select the best language token independently for each decoder row.
    std::vector<int64_t> output_tokens(batch_size);
    for (size_t batch = 0; batch < batch_size; batch++) {
        const float* row_logits = logits_data + (batch * seq_len + (seq_len - 1)) * vocab_size;

        int64_t output_token = -1;
        float max_prob = -std::numeric_limits<float>::infinity();

        for (const auto& [_, lang_token] : config.lang_to_id) {
            auto prob = row_logits[lang_token];
            if (prob > max_prob) {
                max_prob = prob;
                output_token = lang_token;
            }
        }

        output_tokens[batch] = output_token;
    }

    reset_state();

    return {output_tokens, infer_ms};
}

/**
 * Binds encoder hidden states to the decoder request.
 *
 * Supported layouts:
 *  - equal widths: decoder row r attends to encoder row r for scalar or batched greedy decoding;
 *  - one encoder row: the row is broadcast to all decoder rows for single-audio beam search.
 */
void WhisperDecoder::_set_encoder_hidden_states_tensor(const Tensor& encoder_hidden_state,
                                                       const size_t batch_size,
                                                       InferRequest& request) {
    const size_t current_batch_size = request.get_tensor("encoder_hidden_states").get_shape().at(0);
    // A matching width means this round's encoder tensor is already bound; reset_state() clears it between rounds.
    if (current_batch_size == batch_size) {
        return;
    }

    const size_t num_audios = encoder_hidden_state.get_shape().at(0);

    if (num_audios == batch_size) {
        request.set_tensor("encoder_hidden_states", encoder_hidden_state);
        return;
    }

    // Only a single encoder row may be expanded to multiple decoder rows.
    OPENVINO_ASSERT(
        num_audios == 1,
        "Encoder and decoder batch sizes must match unless a single audio is expanded for beam search. Got ",
        num_audios,
        " encoder rows and ",
        batch_size,
        " decoder rows.");

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
