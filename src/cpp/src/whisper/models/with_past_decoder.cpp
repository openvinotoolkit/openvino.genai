// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "with_past_decoder.hpp"

#include <regex>

#include "logger.hpp"
#include "utils.hpp"

namespace {

void copy_with_beam_gather(const ov::Tensor& source, ov::Tensor& dest, const ov::Tensor& beam_idx) {
    const size_t dest_batch_size = beam_idx.get_shape().at(0);

    ov::Shape dest_shape{source.get_shape()};
    dest_shape[0] = dest_batch_size;
    dest.set_shape(dest_shape);

    OPENVINO_ASSERT(dest_shape.size() == 4);

    const size_t batch_dim_size = dest_shape[1] * dest_shape[2] * dest_shape[3];

    const auto beam_idx_data = beam_idx.data<int32_t>();
    const auto source_data = source.data<float>();
    auto dest_data = dest.data<float>();

    for (size_t dest_batch = 0; dest_batch < dest_batch_size; dest_batch++) {
        const size_t source_batch = beam_idx_data[dest_batch];

        const auto source_start = source_data + (source_batch * batch_dim_size);
        const auto dest_start = dest_data + (dest_batch * batch_dim_size);
        std::memcpy(dest_start, source_start, sizeof(float) * batch_dim_size);
    }
}

void copy_past_key_value(ov::InferRequest& source, ov::InferRequest& dest, const ov::Tensor& beam_idx) {
    // source outputs:
    // present.0.decoder.key
    // present.0.decoder.value
    // present.0.encoder.key
    // present.0.encoder.value

    // dest inputs:
    // past_key_values.0.decoder.key
    // past_key_values.0.decoder.value
    // past_key_values.0.encoder.key
    // past_key_values.0.encoder.value

    for (auto& source_output : source.get_compiled_model().outputs()) {
        std::string source_output_name = source_output.get_any_name();
        if (source_output_name.find("present") == std::string::npos) {
            continue;
        }

        std::string dest_input_name = std::regex_replace(source_output_name, std::regex("present"), "past_key_values");

        auto source_tensor = source.get_tensor(source_output_name);
        auto dest_tensor = dest.get_tensor(dest_input_name);

        copy_with_beam_gather(source_tensor, dest_tensor, beam_idx);
    }
}

void link_past_key_value(ov::InferRequest& source, ov::InferRequest& dest) {
    for (auto& source_output : source.get_compiled_model().outputs()) {
        std::string source_output_name = source_output.get_any_name();
        if (source_output_name.find("present") == std::string::npos) {
            continue;
        }

        std::string dest_input_name = std::regex_replace(source_output_name, std::regex("present"), "past_key_values");
        auto source_tensor = source.get_tensor(source_output_name);

        dest.set_tensor(dest_input_name, source_tensor);
    }
}

}  // namespace

namespace ov::genai {
WhisperWithPastDecoder::WhisperWithPastDecoder(const std::filesystem::path& models_path,
                                               const std::string& device,
                                               const ov::AnyMap& properties) {
    Logger::warn("Whisper decoder models with past is deprecated. Support will be removed in 2026.0.0 release.\n"
                 "To obtain stateful decoder model use latest `optimum-intel` package:\n"
                 "pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git@main\n"
                 "optimum-cli export openvino --trust-remote-code --model openai/whisper-tiny whisper-tiny");
    ov::Core core = utils::singleton_core();

    auto compiled_model = core.compile_model(models_path / "openvino_decoder_model.xml", device, properties);
    utils::print_compiled_model_properties(compiled_model, "whisper decoder model");
    m_request_decoder = compiled_model.create_infer_request();

    compiled_model = core.compile_model(models_path / "openvino_decoder_with_past_model.xml", device, properties);
    utils::print_compiled_model_properties(compiled_model, "whisper decoder with past model");
    m_request_decoder_with_past = compiled_model.create_infer_request();
}

void WhisperWithPastDecoder::start_async(const Tensor& encoder_hidden_state,
                                         const Tensor& input_ids,
                                         const Tensor& beam_idx) {
    const bool is_initial_step = m_cache_position == 0;
    ov::InferRequest& request = is_initial_step ? m_request_decoder : m_request_decoder_with_past;

    const size_t batch_size = input_ids.get_shape().at(0);
    const size_t seq_length = input_ids.get_shape().at(1);

    _set_encoder_hidden_states_tensor(encoder_hidden_state, batch_size, request);
    request.set_tensor("input_ids", input_ids);

    if (!is_initial_step) {
        ov::Tensor cache_position_tensor = request.get_tensor("cache_position");
        cache_position_tensor.set_shape({1});
        cache_position_tensor.data<int64_t>()[0] = m_cache_position;
    }

    _set_past_key_value(beam_idx);

    request.start_async();
}

Tensor WhisperWithPastDecoder::wait() {
    const bool is_initial_step = m_cache_position == 0;
    ov::InferRequest& request = is_initial_step ? m_request_decoder : m_request_decoder_with_past;

    request.wait();

    const size_t seq_length = request.get_tensor("input_ids").get_shape().at(1);

    m_cache_position += seq_length;

    return request.get_tensor("logits");
}

void WhisperWithPastDecoder::_set_past_key_value(const Tensor& beam_idx) {
    const bool is_initial_step = m_cache_position == 0;
    if (is_initial_step) {
        return;
    }

    const size_t batch_size = beam_idx.get_shape().at(0);
    // no copy needed, just 'link' output tensor with input tensor
    const bool can_link_past_key_value = batch_size == 1 && beam_idx.data<int32_t>()[0] == 0;

    if (!m_initial_past_key_value_set) {
        if (can_link_past_key_value) {
            link_past_key_value(m_request_decoder, m_request_decoder_with_past);
        } else {
            copy_past_key_value(m_request_decoder, m_request_decoder_with_past, beam_idx);
        }

        m_initial_past_key_value_set = true;
        return;
    }

    if (m_past_key_value_linked) {
        return;
    }

    if (can_link_past_key_value) {
        link_past_key_value(m_request_decoder_with_past, m_request_decoder_with_past);
        m_past_key_value_linked = true;
    } else {
        copy_past_key_value(m_request_decoder_with_past, m_request_decoder_with_past, beam_idx);
    }
};

void WhisperWithPastDecoder::reset_state() {
    m_request_decoder_with_past.reset_state();
    m_cache_position = 0;
    m_initial_past_key_value_set = false;
    m_past_key_value_linked = false;

    Shape encoder_hidden_states_shape{m_request_decoder_with_past.get_tensor("encoder_hidden_states").get_shape()};
    encoder_hidden_states_shape[0] = 0;
    m_request_decoder.set_tensor("encoder_hidden_states", ov::Tensor{ov::element::f32, encoder_hidden_states_shape});
    m_request_decoder_with_past.set_tensor("encoder_hidden_states",
                                           ov::Tensor{ov::element::f32, encoder_hidden_states_shape});
}
}  // namespace ov::genai
