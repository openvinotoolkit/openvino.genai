// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include <chrono>

#include "utils.hpp"

namespace ov::genai {

Qwen3EncoderDecoderASR::Qwen3EncoderDecoderASR(const std::filesystem::path& models_path,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : Qwen3ASRPipelineBase(models_path, properties) {
    ov::AnyMap properties_copy = properties;
    erase_allowed_asr_ctor_properties(properties_copy);
    m_encoder = std::make_unique<Qwen3ASREncoder>(models_path, device, properties_copy);
    size_t max_encoder_len = 0;
    if (device == "NPU") {
        m_max_asr_input_seconds = 60;
        max_encoder_len = m_encoder->get_npuw_qwen3_asr_max_encoder_len(m_max_asr_input_seconds,
                                                                      m_feature_extractor.sampling_rate,
                                                                      m_feature_extractor.hop_length);
    }
    m_decoder = std::make_unique<Qwen3ASRDecoder>(models_path, device, properties_copy, max_encoder_len);
    m_decoder->set_seed(m_generation_config.rng_seed);
}

std::string Qwen3EncoderDecoderASR::extend_audio_tokens(const std::string& prompt, size_t audio_length) const {
    static const std::string audio_token = "<|audio_pad|>";

    const size_t token_pos = prompt.find(audio_token);
    OPENVINO_ASSERT(token_pos != std::string::npos, "extend_audio_tokens: audio token not found in prompt");

    std::string replacement;
    replacement.reserve(audio_token.size() * audio_length);
    for (size_t j = 0; j < audio_length; ++j) {
        replacement += audio_token;
    }

    std::string result = prompt;
    result.replace(token_pos, audio_token.size(), replacement);
    return result;
}

std::vector<std::string> Qwen3EncoderDecoderASR::infer(const std::vector<AudioChunk>& chunks,
                                                       const ASRGenerationConfig& config,
                                                       ASRPerfMetrics& perf_metrics,
                                                       const std::shared_ptr<StreamerBase>& streamer) {
    m_decoder->set_seed(config.rng_seed);

    // inference batch_size. Can be different to input batch size due to chunking
    const size_t batch_size = chunks.size();
    const std::string prompt = build_text_prompt(config);

    std::vector<WhisperFeatures> features;
    features.reserve(batch_size);
    const auto features_start_time = std::chrono::steady_clock::now();
    for (const auto& chunk : chunks) {
        features.emplace_back(m_feature_extractor.extract(chunk.wav, false));
    }
    const auto features_stop_time = std::chrono::steady_clock::now();
    perf_metrics.asr_raw_metrics.features_extraction_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(features_stop_time - features_start_time)));

    std::vector<std::string> results;
    results.reserve(batch_size);

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const auto encoder_start_time = std::chrono::steady_clock::now();
        const ov::Tensor encoder_hidden_states = m_encoder->encode(features[batch]);
        const auto encoder_stop_time = std::chrono::steady_clock::now();
        const auto encoder_infer_ms = PerfMetrics::get_microsec(encoder_stop_time - encoder_start_time);
        perf_metrics.raw_metrics.m_inference_durations[0] += MicroSeconds(encoder_infer_ms);
        perf_metrics.asr_raw_metrics.encode_inference_durations.emplace_back(encoder_infer_ms);

        const size_t audio_token_count = encoder_hidden_states.get_shape()[1];

        const std::string processed_prompt = extend_audio_tokens(prompt, audio_token_count);

        const auto tokenization_start_time = std::chrono::steady_clock::now();
        const ov::Tensor input_ids = m_tokenizer.encode(processed_prompt).input_ids;
        const auto tokenization_stop_time = std::chrono::steady_clock::now();
        perf_metrics.raw_metrics.tokenization_durations.emplace_back(
            MicroSeconds(PerfMetrics::get_microsec(tokenization_stop_time - tokenization_start_time)));

        const auto encoded_results = m_decoder->generate(input_ids,
                                                         encoder_hidden_states,
                                                         config,
                                                         perf_metrics.raw_metrics,
                                                         perf_metrics.asr_raw_metrics,
                                                         create_chunk_streamer(streamer, config));

        const auto detokenization_start_time = std::chrono::steady_clock::now();
        const auto text = m_tokenizer.decode(encoded_results.tokens[0]);
        const auto detokenization_stop_time = std::chrono::steady_clock::now();
        perf_metrics.raw_metrics.detokenization_durations.emplace_back(
            MicroSeconds(PerfMetrics::get_microsec(detokenization_stop_time - detokenization_start_time)));
        results.push_back(text);
    }

    return results;
}

}  // namespace ov::genai
