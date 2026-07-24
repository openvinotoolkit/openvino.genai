// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include <algorithm>
#include <chrono>

#include "audio_chunk.hpp"
#include "streamer.hpp"
#include "utils.hpp"

namespace {

int64_t get_required_token_id(const ov::genai::Tokenizer& tokenizer, const std::string& token) {
    const ov::genai::Vocab vocab = tokenizer.get_vocab();
    const auto token_it = vocab.find(token);
    OPENVINO_ASSERT(token_it != vocab.end(), "Qwen3-ASR tokenizer must contain '", token, "' token");
    return token_it->second;
}

}  // namespace

namespace ov::genai {

Qwen3ASR::Qwen3ASR(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties)
    : ASRPipelineImplBase(models_path),
      m_feature_extractor{models_path / "preprocessor_config.json"},
      m_asr_text_token_id{get_required_token_id(m_tokenizer, "<asr_text>")} {
    ov::AnyMap properties_copy = properties;
    erase_allowed_asr_ctor_properties(properties_copy);
    const auto encoder_properties = utils::get_model_properties(properties_copy, "audio_encoder", device);
    m_encoder =
        std::make_unique<Qwen3ASREncoder>(models_path, device, encoder_properties, m_feature_extractor.feature_size);
    m_decoder = std::make_unique<Qwen3ASRDecoder>(models_path, device, properties_copy);

    // Qwen3-ASR EOS tokens: <|endoftext|>=151643, <|im_end|>=151645
    // The exported model has no generation_config.json. Qwen3-ASR original implementation hardcodes them as well.
    m_generation_config.set_eos_token_id(151643);
    m_generation_config.stop_token_ids.insert(151645);
    m_decoder->set_seed(m_generation_config.rng_seed);
}

ASRDecodedResults Qwen3ASR::generate(const AudioInputs& audio_inputs,
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

    const std::vector<AudioChunk> chunks =
        split_audio_into_chunks({audio}, m_feature_extractor.sampling_rate, MAX_ASR_INPUT_SECONDS);

    const auto infer_results = infer(chunks, config, results.perf_metrics, streamer);
    if (streamer) {
        streamer->end();
    }

    auto [merged_texts, merged_languages] = merge_chunk_results(chunks, infer_results, config);
    results.languages = std::move(merged_languages);
    for (size_t i = 0; i < merged_texts.size(); ++i) {
        results.texts.push_back(merged_texts[i]);
        results.scores.push_back(0.0f);
    }

    auto stop_time = std::chrono::steady_clock::now();
    results.perf_metrics.raw_metrics.generate_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(stop_time - start_time)));
    results.perf_metrics.evaluate_statistics(start_time);
    return results;
}

std::vector<std::string> Qwen3ASR::build_text_prompt(size_t batch_size, const ASRGenerationConfig& config) {
    std::string context;
    if (config.context.has_value()) {
        context = config.context.value();
    }

    // Hardcoded chat template to avoid tool-call warnings.
    std::string prompt = "<|im_start|>system\n" + context +
                         "<|im_end|>\n"
                         "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
                         "<|im_start|>assistant\n";

    if (config.language.has_value() && !config.language.value().empty()) {
        prompt += "language " + config.language.value() + "<asr_text>";
    }

    return std::vector<std::string>(batch_size, prompt);
}

std::pair<std::string, std::string> Qwen3ASR::parse_asr_output(const std::string& raw,
                                                               const std::optional<std::string>& forced_language) {
    static const std::string asr_text_tag = "<asr_text>";
    static const std::string lang_prefix = "language ";

    if (forced_language.has_value() && !forced_language.value().empty()) {
        return {forced_language.value(), raw};
    }

    const size_t tag_pos = raw.find(asr_text_tag);
    if (tag_pos == std::string::npos) {
        return {"", raw};
    }

    const std::string meta_part = raw.substr(0, tag_pos);
    const std::string text_part = raw.substr(tag_pos + asr_text_tag.size());

    // empty audio heuristic: "language None<asr_text>"
    if (meta_part.find("language None") != std::string::npos || meta_part.find("language none") != std::string::npos) {
        return {"", text_part};
    }

    // extract language name from "language <Name>"
    const size_t lang_pos = meta_part.find(lang_prefix);
    if (lang_pos == std::string::npos) {
        return {"", text_part};
    }
    const std::string language = meta_part.substr(lang_pos + lang_prefix.size());

    return {language, text_part};
}

std::pair<std::vector<std::string>, std::vector<std::string>> Qwen3ASR::merge_chunk_results(
    const std::vector<AudioChunk>& chunks,
    const std::vector<std::string>& infer_results,
    const ASRGenerationConfig& config) {
    size_t num_samples = 0;
    for (const auto& chunk : chunks) {
        num_samples = std::max(num_samples, chunk.orig_batch + 1);
    }

    std::vector<std::string> merged_texts(num_samples);
    std::vector<std::string> merged_languages(num_samples);
    for (size_t i = 0; i < chunks.size(); ++i) {
        const size_t orig = chunks[i].orig_batch;
        auto [language, text] = parse_asr_output(infer_results[i], config.language);
        if (!merged_texts[orig].empty()) {
            merged_texts[orig] += ' ';
        }
        merged_texts[orig] += text;
        merged_languages[orig] = std::move(language);  // last wins
    }

    return {std::move(merged_texts), std::move(merged_languages)};
}

std::vector<std::string> Qwen3ASR::extend_audio_tokens(const std::vector<std::string>& prompts,
                                                       const std::vector<size_t>& audio_lengths) {
    OPENVINO_ASSERT(prompts.size() == audio_lengths.size(),
                    "extend_audio_tokens: prompts and audio_lengths must have the same size");

    static const std::string audio_token = "<|audio_pad|>";

    std::vector<std::string> results = prompts;

    for (size_t i = 0; i < prompts.size(); ++i) {
        const size_t token_pos = prompts[i].find(audio_token);
        OPENVINO_ASSERT(token_pos != std::string::npos, "extend_audio_tokens: audio token not found in prompt");

        std::string replacement;
        replacement.reserve(audio_token.size() * audio_lengths[i]);
        for (size_t j = 0; j < audio_lengths[i]; ++j) {
            replacement += audio_token;
        }

        results[i].replace(token_pos, audio_token.size(), replacement);
    }

    return results;
}

std::vector<std::string> Qwen3ASR::infer(std::vector<AudioChunk> chunks,
                                         const ASRGenerationConfig& config,
                                         ASRPerfMetrics& perf_metrics,
                                         const std::shared_ptr<StreamerBase>& streamer_ptr) {
    // inference batch_size. Can be different to input batch size due to chunking
    const size_t batch_size = chunks.size();
    const std::vector<std::string> prompts = build_text_prompt(batch_size, config);

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

    const bool forced_language = config.language.has_value() && !config.language.value().empty();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const std::string prompt = prompts[batch];
        const auto encoder_start_time = std::chrono::steady_clock::now();
        const ov::Tensor encoder_hidden_states = m_encoder->encode(features[batch]);
        const auto encoder_stop_time = std::chrono::steady_clock::now();
        const auto encoder_infer_ms = PerfMetrics::get_microsec(encoder_stop_time - encoder_start_time);
        perf_metrics.raw_metrics.m_inference_durations[0] += MicroSeconds(encoder_infer_ms);
        perf_metrics.asr_raw_metrics.encode_inference_durations.emplace_back(encoder_infer_ms);

        const size_t audio_token_count = encoder_hidden_states.get_shape()[1];

        const std::vector<std::string> processed_prompts = extend_audio_tokens({prompt}, {audio_token_count});

        const auto tokenization_start_time = std::chrono::steady_clock::now();
        const ov::Tensor input_ids = m_tokenizer.encode(processed_prompts).input_ids;
        const auto tokenization_stop_time = std::chrono::steady_clock::now();
        perf_metrics.raw_metrics.tokenization_durations.emplace_back(
            MicroSeconds(PerfMetrics::get_microsec(tokenization_stop_time - tokenization_start_time)));

        // streamer wrapper should be reset for each batch to properly handle suppression
        const std::shared_ptr<StreamerBase> decoder_streamer =
            streamer_ptr && !forced_language
                ? std::make_shared<Qwen3ASRStreamer>(streamer_ptr, m_asr_text_token_id, true)
                : streamer_ptr;

        const auto encoded_results = m_decoder->generate(input_ids,
                                                         encoder_hidden_states,
                                                         config,
                                                         perf_metrics.raw_metrics,
                                                         perf_metrics.asr_raw_metrics,
                                                         decoder_streamer);

        const auto detokenization_start_time = std::chrono::steady_clock::now();
        const auto text = m_tokenizer.decode(encoded_results.tokens[0]);
        const auto detokenization_stop_time = std::chrono::steady_clock::now();
        perf_metrics.raw_metrics.detokenization_durations.emplace_back(
            MicroSeconds(PerfMetrics::get_microsec(detokenization_stop_time - detokenization_start_time)));
        results.push_back(text);
    }

    return results;
}

ASRGenerationConfig Qwen3ASR::resolve_generation_config(const std::optional<ASRGenerationConfig>& generation_config) const {
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

void Qwen3ASR::validate_generation_config(const ASRGenerationConfig& config) const {
    config.validate();

    OPENVINO_ASSERT(!config.is_beam_search(), "Qwen3-ASR does not support beam search decoding");
}

}  // namespace ov::genai
