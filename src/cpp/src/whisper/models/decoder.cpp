// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <numeric>

#include "cross_kv_decoder.hpp"
#include "statefull_decoder.hpp"
#include "whisper/whisper_utils.hpp"

namespace {

int64_t most_likely_language(const float* logits, const ov::genai::WhisperGenerationConfig& config) {
    int64_t language_token = -1;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (const auto& [_, token] : config.lang_to_id) {
        if (logits[token] > max_logit) {
            max_logit = logits[token];
            language_token = token;
        }
    }
    return language_token;
}

}  // namespace

namespace ov::genai {
std::shared_ptr<WhisperDecoder> WhisperDecoder::from_path(const std::filesystem::path& models_path,
                                                          const std::string& device,
                                                          const ov::AnyMap& properties,
                                                          const ov::PartialShape& lhs_shape,
                                                          const bool decompose_cross_attention_spda_ops,
                                                          const ov::element::Type& inference_precision) {
    if (device == "CPU" && is_batch_validated_precision(inference_precision)) {
        if (auto decoder = WhisperCrossKVDecoder::create(models_path,
                                                         device,
                                                         properties,
                                                         lhs_shape,
                                                         decompose_cross_attention_spda_ops)) {
            return decoder;
        }
    }
    return std::make_shared<WhisperStatefullDecoder>(models_path,
                                                     device,
                                                     properties,
                                                     lhs_shape,
                                                     decompose_cross_attention_spda_ops);
}

std::pair<int64_t, float> WhisperDecoder::detect_language(const ov::Tensor& encoder_hidden_state,
                                                          const WhisperGenerationConfig& config) {
    auto [logits, infer_ms] = infer_language_logits(encoder_hidden_state, config, 1);
    const int64_t language_token = most_likely_language(logits.data<float>(), config);

    reset_state();

    return {language_token, infer_ms};
}

std::pair<std::vector<int64_t>, float> WhisperDecoder::detect_languages(const ov::Tensor& encoder_hidden_state,
                                                                        const WhisperGenerationConfig& config) {
    const size_t batch_size = encoder_hidden_state.get_shape().at(0);
    OPENVINO_ASSERT(batch_size > 0, "Language detection requires at least one encoder hidden state row.");
    OPENVINO_ASSERT(batch_size == 1 || supports_dynamic_batch(),
                    "This Whisper decoder detects the language of one audio at a time, got ",
                    batch_size,
                    ".");

    auto [logits, infer_ms] = infer_language_logits(encoder_hidden_state, config, batch_size);

    const auto logits_shape = logits.get_shape();
    const size_t seq_len = logits_shape.at(1);
    const size_t vocab_size = logits_shape.at(2);
    const auto* logits_data = logits.data<float>();

    std::vector<int64_t> language_tokens;
    language_tokens.reserve(batch_size);
    for (size_t batch = 0; batch < batch_size; batch++) {
        language_tokens.push_back(
            most_likely_language(logits_data + (batch * seq_len + (seq_len - 1)) * vocab_size, config));
    }

    reset_state();

    return {language_tokens, infer_ms};
}

std::pair<ov::Tensor, float> WhisperDecoder::infer_language_logits(const ov::Tensor& encoder_hidden_state,
                                                                   const WhisperGenerationConfig& config,
                                                                   const size_t rows) {
    Tensor input_ids_tensor = create_host_tensor(ov::element::i64, {rows, 1});
    std::fill_n(input_ids_tensor.data<int64_t>(), rows, config.decoder_start_token_id);

    Tensor beam_idx_tensor = create_host_tensor(ov::element::i32, {rows});
    std::iota(beam_idx_tensor.data<int32_t>(), beam_idx_tensor.data<int32_t>() + rows, 0);

    const auto infer_start = std::chrono::steady_clock::now();
    start_async(encoder_hidden_state, input_ids_tensor, beam_idx_tensor);

    Tensor logits = wait();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

    return {logits, infer_ms};
}

ov::Tensor WhisperDecoder::create_host_tensor(const element::Type element_type, const Shape& shape) {
    return ov::Tensor(element_type, shape);
}

WhisperDecoder::~WhisperDecoder() = default;
}  // namespace ov::genai
