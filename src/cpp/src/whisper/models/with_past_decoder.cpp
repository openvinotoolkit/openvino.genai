// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "with_past_decoder.hpp"

#include <regex>

#include "logger.hpp"
#include "utils.hpp"

namespace {
void set_past_key_value(ov::InferRequest& source, ov::InferRequest& dest) {
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
        if (source_output_name.find("logits") != std::string::npos) {
            continue;
        }

        std::string with_past_input_name =
            std::regex_replace(source_output_name, std::regex("present"), "past_key_values");

        auto kv_tensor = source.get_tensor(source_output_name);
        dest.set_tensor(with_past_input_name, ov::Tensor{kv_tensor});
    }
}
}  // namespace

namespace ov::genai {
WhisperWithPastDecoder::WhisperWithPastDecoder(const std::filesystem::path& models_path,
                                               const std::string& device,
                                               const ov::AnyMap& properties) {
    Logger::warn("Whisper decoder models with past is deprecated. Support will be removed in 2026.0.0 release.\n"
                 "To obtain stateful decoder model use latest `optimum-intel` package:\n"
                 "pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git\n"
                 "optimum-cli export openvino --trust-remote-code --model openai/whisper-tiny whisper-tiny");
    ov::Core core = utils::singleton_core();

    auto compiled_model = core.compile_model(models_path / "openvino_decoder_model.xml", device, properties);
    utils::print_compiled_model_properties(compiled_model, "whisper decoder model");
    m_request_decoder = compiled_model.create_infer_request();

    compiled_model = core.compile_model(models_path / "openvino_decoder_with_past_model.xml", device, properties);
    utils::print_compiled_model_properties(compiled_model, "whisper decoder with past model");
    m_request_decoder_with_past = compiled_model.create_infer_request();
}

std::pair<int64_t, float> WhisperWithPastDecoder::detect_language(const ov::Tensor& encoder_hidden_state,
                                                                  const int64_t decoder_start_token_id) {
    auto [output_tensor, infer_ms] = decode(encoder_hidden_state, {decoder_start_token_id}, 0);

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    reset_state();

    return {output_token, infer_ms};
}

std::pair<ov::Tensor, float> WhisperWithPastDecoder::decode(const ov::Tensor& encoder_hidden_state,
                                                            const std::vector<int64_t>& input_ids,
                                                            const size_t cache_position) {
    const bool initial_step = cache_position == 0;
    ov::InferRequest& request = initial_step ? m_request_decoder : m_request_decoder_with_past;

    request.set_tensor("encoder_hidden_states", encoder_hidden_state);

    const ov::Tensor input_ids_tensor(ov::element::i64, {1, input_ids.size()}, (void*)input_ids.data());
    request.set_tensor("input_ids", input_ids_tensor);

    if (!initial_step) {
        ov::Tensor cache_position_tensor = request.get_tensor("cache_position");
        cache_position_tensor.set_shape({1});
        cache_position_tensor.data<int64_t>()[0] = cache_position;
    }

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    auto output_tensor = request.get_tensor("logits");

    if (initial_step) {
        set_past_key_value(m_request_decoder, m_request_decoder_with_past);
    } else if (!m_decoder_with_past_kv_value_set) {
        set_past_key_value(m_request_decoder_with_past, m_request_decoder_with_past);
        m_decoder_with_past_kv_value_set = true;
    }

    return {output_tensor, infer_ms};
}

void WhisperWithPastDecoder::reset_state() {
    m_request_decoder_with_past.reset_state();
    m_decoder_with_past_kv_value_set = false;
}
}  // namespace ov::genai
