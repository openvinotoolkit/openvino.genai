// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "statefull_decoder.hpp"

#include "utils.hpp"

namespace ov::genai {
WhisperStatefullDecoder::WhisperStatefullDecoder(const std::filesystem::path& models_path,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();

    auto model = core.read_model((models_path / "openvino_decoder_model.xml").string());

    // todo: remove once stateful model has dynamic input_ids seq_len
    std::map<std::string, ov::PartialShape> name_to_shape;
    for (const ov::Output<ov::Node>& input : model->inputs()) {
        ov::PartialShape shape = input.get_partial_shape();
        if (input.get_any_name().find("input_ids") != std::string::npos) {
            shape[1] = -1;
            name_to_shape[input.get_any_name()] = shape;
        }
    }
    model->reshape(name_to_shape);

    auto compiled_model = core.compile_model(model, device, properties);

    utils::print_compiled_model_properties(compiled_model, "whisper decoder model");
    m_request = compiled_model.create_infer_request();
}

std::pair<int64_t, float> WhisperStatefullDecoder::detect_language(const ov::Tensor& encoder_hidden_state,
                                                                   const int64_t decoder_start_token_id) {
    auto [output_tensor, infer_ms] = decode(encoder_hidden_state, {decoder_start_token_id}, 0);

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    reset_state();

    return {output_token, infer_ms};
}

std::pair<ov::Tensor, float> WhisperStatefullDecoder::decode(const ov::Tensor& encoder_hidden_state,
                                                             const std::vector<int64_t>& input_ids,
                                                             const size_t cache_position) {
    m_request.set_tensor("encoder_hidden_states", encoder_hidden_state);

    ov::Tensor input_ids_tensor(ov::element::i64, {1, input_ids.size()}, (void*)input_ids.data());
    m_request.set_tensor("input_ids", input_ids_tensor);

    ov::Tensor cache_position_tensor = m_request.get_tensor("cache_position");
    cache_position_tensor.set_shape({input_ids.size()});

    auto cache_data = cache_position_tensor.data<int64_t>();
    std::iota(cache_data, cache_data + cache_position_tensor.get_size(), cache_position);

    m_request.get_tensor("beam_idx").set_shape({1});
    m_request.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    const auto infer_start = std::chrono::steady_clock::now();
    m_request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    auto output_tensor = m_request.get_tensor("logits");

    return {output_tensor, infer_ms};
};

void WhisperStatefullDecoder::reset_state() {
    m_request.reset_state();
}
}  // namespace ov::genai
