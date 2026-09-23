// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include <chrono>
#include <cstring>

#include "utils.hpp"

namespace {

struct FunASRTextPrompt {
    std::string prefix;
    std::string suffix;
    std::string language;
};

FunASRTextPrompt build_text_prompt(const ov::genai::ASRGenerationConfig& config) {
    const std::string language_instruction = config.language.has_value() ? "语音转写成" + *config.language : "语音转写";

    // "\xEF\xBC\x9A" - UTF-8 full-width colon "：". ASCII ":" gives worse accuracy.
    const std::string prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" +
                               language_instruction + "\xEF\xBC\x9A";
    const std::string suffix = "<|im_end|>\n<|im_start|>assistant\n";
    // Fun-ASR does not detect the spoken language, so return the requested language or an empty string.
    return {prefix, suffix, config.language.value_or("")};
}

}  // namespace

namespace ov::genai {

FunASR::FunASR(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties)
    : ASRPipelineImplBase(models_path) {
    ov::AnyMap properties_copy = properties;
    erase_allowed_asr_ctor_properties(properties_copy);
    m_encoder = std::make_unique<FunASREncoder>(models_path, device, properties_copy);
    m_decoder = std::make_unique<Qwen3ASRDecoder>(models_path, device, properties_copy);

    // Qwen3-ASR EOS tokens: <|endoftext|>=151643, <|im_end|>=151645
    // The exported model has no generation_config.json. Hardcode tokens. Algned with Qwen3-ASR
    m_generation_config.set_eos_token_id(151643);
    m_generation_config.stop_token_ids.insert(151645);
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

    auto [prompt, language] =
        build_input_ids(encoder_hidden_states.get_shape()[1], config, results.perf_metrics.raw_metrics);

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
    results.languages.push_back(std::move(language));
    results.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(detokenization_stop_time - detokenization_start_time)));

    auto stop_time = std::chrono::steady_clock::now();
    results.perf_metrics.raw_metrics.generate_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(stop_time - start_time)));
    results.perf_metrics.evaluate_statistics(start_time);
    return results;
}

std::pair<ov::Tensor, std::string> FunASR::build_input_ids(const size_t num_audio_tokens,
                                                           const ASRGenerationConfig& config,
                                                           RawPerfMetrics& raw_metrics) {
    FunASRTextPrompt text_prompt = build_text_prompt(config);
    const auto tokenization_start_time = std::chrono::steady_clock::now();
    const ov::Tensor prefix_ids =
        m_tokenizer.encode(text_prompt.prefix, ov::genai::add_special_tokens(false)).input_ids;
    const ov::Tensor suffix_ids =
        m_tokenizer.encode(text_prompt.suffix, ov::genai::add_special_tokens(false)).input_ids;
    const auto tokenization_stop_time = std::chrono::steady_clock::now();
    raw_metrics.tokenization_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(tokenization_stop_time - tokenization_start_time)));

    const size_t prompt_length = prefix_ids.get_size() + num_audio_tokens + suffix_ids.get_size();

    ov::Tensor input_ids(ov::element::i64, {1, prompt_length});
    int64_t* input_ids_data = input_ids.data<int64_t>();
    std::memcpy(input_ids_data, prefix_ids.data<const int64_t>(), prefix_ids.get_byte_size());
    input_ids_data += prefix_ids.get_size();
    std::fill_n(input_ids_data, num_audio_tokens, 0);
    input_ids_data += num_audio_tokens;
    std::memcpy(input_ids_data, suffix_ids.data<const int64_t>(), suffix_ids.get_byte_size());

    return {input_ids, std::move(text_prompt.language)};
}

ASRGenerationConfig FunASR::resolve_generation_config(
    const std::optional<ASRGenerationConfig>& generation_config) const {
    ASRGenerationConfig config = generation_config.value_or(m_generation_config);
    if (config.stop_token_ids.empty()) {
        config.stop_token_ids = m_generation_config.stop_token_ids;
    }

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
