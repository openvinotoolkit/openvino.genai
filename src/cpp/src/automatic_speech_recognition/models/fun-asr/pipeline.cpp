// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include <chrono>
#include <cstring>

#include "utils.hpp"

namespace ov::genai {

FunASR::FunASR(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties)
    : ASRPipelineImplBase(models_path) {
    ov::AnyMap properties_copy = properties;
    erase_allowed_asr_ctor_properties(properties_copy);
    m_encoder = std::make_unique<FunASREncoder>(models_path, device, properties_copy);
    m_decoder = std::make_unique<Qwen3ASRDecoder>(models_path, device, properties_copy);
    m_generation_config.set_eos_token_id(151645);
    m_decoder->set_seed(m_generation_config.rng_seed);
}

ASRDecodedResults FunASR::generate(const AudioInputs& audio_inputs,
                                   const std::optional<ASRGenerationConfig>& generation_config,
                                   const std::shared_ptr<StreamerBase> streamer) {
    auto start_time = std::chrono::steady_clock::now();

    const ASRGenerationConfig config = resolve_generation_config(generation_config);
    m_decoder->set_seed(config.rng_seed);

    ASRDecodedResults results;
    results.perf_metrics.load_time = m_load_time_ms;
    results.perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    const std::vector<float>& audio = std::visit(
        ov::genai::utils::overloaded{
            [](const std::vector<float>& input) -> const std::vector<float>& {
                return input;
            },
        },
        audio_inputs);

    const auto features_start_time = std::chrono::steady_clock::now();
    const ov::Tensor features = m_feature_extractor.extract(audio);
    const auto features_stop_time = std::chrono::steady_clock::now();
    results.perf_metrics.asr_raw_metrics.features_extraction_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(features_stop_time - features_start_time)));

    const auto encoder_start_time = std::chrono::steady_clock::now();
    const ov::Tensor encoder_hidden_states = m_encoder->encode(features);
    const auto encoder_stop_time = std::chrono::steady_clock::now();
    const auto encoder_infer_ms = PerfMetrics::get_microsec(encoder_stop_time - encoder_start_time);
    results.perf_metrics.raw_metrics.m_inference_durations[0] += MicroSeconds(encoder_infer_ms);
    results.perf_metrics.asr_raw_metrics.encode_inference_durations.emplace_back(encoder_infer_ms);

    const auto tokenization_start_time = std::chrono::steady_clock::now();
    const ov::Tensor prompt = build_input_ids(encoder_hidden_states.get_shape()[1], config);
    const auto tokenization_stop_time = std::chrono::steady_clock::now();
    results.perf_metrics.raw_metrics.tokenization_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(tokenization_stop_time - tokenization_start_time)));

    const auto encoded_results = m_decoder->generate(prompt,
                                                     encoder_hidden_states,
                                                     config,
                                                     results.perf_metrics.raw_metrics,
                                                     results.perf_metrics.asr_raw_metrics,
                                                     streamer);
    if (streamer) {
        streamer->end();
    }

    const auto detokenization_start_time = std::chrono::steady_clock::now();
    results.texts.push_back(m_tokenizer.decode(encoded_results.tokens[0]));
    const auto detokenization_stop_time = std::chrono::steady_clock::now();
    results.scores.push_back(encoded_results.scores[0]);
    results.languages.push_back(config.language.value_or(""));
    results.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(detokenization_stop_time - detokenization_start_time)));

    auto stop_time = std::chrono::steady_clock::now();
    results.perf_metrics.raw_metrics.generate_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(stop_time - start_time)));
    results.perf_metrics.evaluate_statistics(start_time);
    return results;
}

ov::Tensor FunASR::build_input_ids(const size_t num_audio_tokens, const ASRGenerationConfig& config) {
    const TokenizedInstructions instructions = get_tokenized_instructions(config);
    const auto prefix_ids = instructions.prefix_ids;
    const auto suffix_ids = instructions.suffix_ids;

    const size_t prompt_length = prefix_ids.get_size() + num_audio_tokens + suffix_ids.get_size();

    ov::Tensor input_ids(ov::element::i64, {1, prompt_length});
    int64_t* input_ids_data = input_ids.data<int64_t>();
    std::memcpy(input_ids_data, prefix_ids.data<const int64_t>(), prefix_ids.get_byte_size());
    input_ids_data += prefix_ids.get_size();
    std::fill_n(input_ids_data, num_audio_tokens, 0);
    input_ids_data += num_audio_tokens;
    std::memcpy(input_ids_data, suffix_ids.data<const int64_t>(), suffix_ids.get_byte_size());

    return input_ids;
}

FunASR::TokenizedInstructions FunASR::get_tokenized_instructions(const ASRGenerationConfig& config) {
    const std::string language = config.language.value_or("中文");
    std::lock_guard<std::mutex> lock(m_tokenized_instructions_mutex);
    const auto cached = m_tokenized_instructions.find(language);
    if (cached != m_tokenized_instructions.end()) {
        return cached->second;
    }

    const std::string prefix =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写成" + language +
        // utf-8 full-width column "：". ascii ":" gives worse accuracy
        "\xEF\xBC\x9A";
    const std::string suffix = "<|im_end|>\n<|im_start|>assistant\n";
    TokenizedInstructions instructions{
        m_tokenizer.encode(prefix, ov::genai::add_special_tokens(false)).input_ids,
        m_tokenizer.encode(suffix, ov::genai::add_special_tokens(false)).input_ids,
    };
    m_tokenized_instructions.emplace(language, instructions);
    return instructions;
}

ASRGenerationConfig FunASR::resolve_generation_config(
    const std::optional<ASRGenerationConfig>& generation_config) const {
    ASRGenerationConfig config = generation_config.value_or(m_generation_config);
    if (config.eos_token_id == -1) {
        config.set_eos_token_id(m_generation_config.eos_token_id);
    }

    validate_generation_config(config);
    return config;
}

void FunASR::validate_generation_config(const ASRGenerationConfig& config) const {
    config.validate();

    OPENVINO_ASSERT(!config.is_beam_search(), "Fun-ASR does not support beam search decoding");
}

}  // namespace ov::genai
