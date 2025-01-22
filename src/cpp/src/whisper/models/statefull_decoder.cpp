// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "statefull_decoder.hpp"

#include "debug_utils.hpp"
#include "utils.hpp"

namespace ov::genai {
WhisperStatefullDecoder::WhisperStatefullDecoder(const std::filesystem::path& models_path,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();

    auto compiled_model = core.compile_model(models_path / "openvino_decoder_model.xml", device, properties);

    utils::print_compiled_model_properties(compiled_model, "whisper decoder model");
    m_request = compiled_model.create_infer_request();
}

std::pair<int64_t, float> WhisperStatefullDecoder::detect_language(const ov::Tensor& encoder_hidden_state,
                                                                   const int64_t decoder_start_token_id) {
    Tensor input_ids_tensor{ov::element::i64, {1, 1}};
    input_ids_tensor.data<int64_t>()[0] = decoder_start_token_id;

    Tensor beam_idx_tensor{ov::element::i32, {1}};
    beam_idx_tensor.data<int32_t>()[0] = 0;

    auto [output_tensor, infer_ms] = decode(encoder_hidden_state, input_ids_tensor, beam_idx_tensor);

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    reset_state();

    return {output_token, infer_ms};
}

std::pair<ov::Tensor, float> WhisperStatefullDecoder::decode(const Tensor& encoder_hidden_state,
                                                             const Tensor& input_ids,
                                                             const Tensor& beam_idx) {
    const size_t batch_size = input_ids.get_shape().at(0);
    const size_t seq_len = input_ids.get_shape().at(1);

    // todo: skip copy if already set and batch didn't changed
    _set_encoder_hidden_states_tensor(encoder_hidden_state, batch_size);

    _set_cache_position_tensor(seq_len);
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("beam_idx", beam_idx);

    const auto infer_start = std::chrono::steady_clock::now();
    m_request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    auto output_tensor = m_request.get_tensor("logits");

    return {output_tensor, infer_ms};
};

/**
 * Encoder hidden states expected to be with batch 1
 * Copy encoder hidden state tensor from batch 1 to requested batch_size.
 * Set new encoder hidden states tensor to infer request.
 */
void WhisperStatefullDecoder::_set_encoder_hidden_states_tensor(const Tensor& encoder_hidden_state,
                                                                const size_t batch_size) {
    _reset_encoder_past_key_values_states(encoder_hidden_state, batch_size);

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

    m_request.set_tensor("encoder_hidden_states", new_encoder_hidden_states);
}

// Ensure encoder past_key values states are reset if batch size changed. This is workaround for Ticket:
void WhisperStatefullDecoder::_reset_encoder_past_key_values_states(const Tensor& encoder_hidden_state,
                                                                    const size_t batch_size) {
    const size_t current_batch_size = m_request.get_tensor("encoder_hidden_states").get_shape().at(0);
    // batch hasn't changed, skip
    if (current_batch_size == 0 || current_batch_size == batch_size) {
        return;
    }

    const size_t encoder_state_length_dim = encoder_hidden_state.get_shape().at(1);
    for (auto& state : m_request.query_state()) {
        // find encoder states by dimension
        const Shape& state_shape = state.get_state().get_shape();
        if (state_shape.at(2) == encoder_state_length_dim) {
            state.reset();
        }
    }
}

void WhisperStatefullDecoder::_set_cache_position_tensor(const size_t seq_len) {
    ov::Tensor cache_position_tensor = m_request.get_tensor("cache_position");

    int64_t start_cache_position = 0;

    if (cache_position_tensor.get_size() != 0) {
        start_cache_position = cache_position_tensor.data<int64_t>()[cache_position_tensor.get_size() - 1] + 1;
    }

    cache_position_tensor.set_shape({seq_len});

    auto cache_data = cache_position_tensor.data<int64_t>();
    std::iota(cache_data, cache_data + seq_len, start_cache_position);
};

void WhisperStatefullDecoder::reset_state() {
    m_request.reset_state();
    m_request.set_tensor("cache_position", ov::Tensor{ov::element::i64, {0}});
};

}  // namespace ov::genai
