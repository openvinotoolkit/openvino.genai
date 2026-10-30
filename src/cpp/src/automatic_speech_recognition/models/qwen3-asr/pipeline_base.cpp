// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/models/qwen3-asr/pipeline_base.hpp"

#include <algorithm>
#include <chrono>
#include <iterator>

#include "automatic_speech_recognition/models/qwen3-asr/streamer.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "utils.hpp"

namespace {

int64_t get_required_token_id(const ov::genai::Tokenizer& tokenizer, const std::string& token) {
    const ov::genai::Vocab vocab = tokenizer.get_vocab();
    const auto token_it = vocab.find(token);
    OPENVINO_ASSERT(token_it != vocab.end(), "Qwen3-ASR tokenizer must contain '", token, "' token");
    return token_it->second;
}

std::shared_ptr<ov::genai::ASRForcedAligner> get_forced_aligner(const ov::AnyMap& properties) {
    if (const auto it = properties.find(ov::genai::forced_aligner.name()); it != properties.end()) {
        return it->second.as<std::shared_ptr<ov::genai::ASRForcedAligner>>();
    }
    return nullptr;
}

size_t get_num_samples(const std::vector<ov::genai::AudioChunk>& chunks) {
    size_t num_samples = 0;
    for (const auto& chunk : chunks) {
        num_samples = std::max(num_samples, chunk.orig_batch + 1);
    }
    return num_samples;
}

}  // namespace

namespace ov::genai {

Qwen3ASRPipelineBase::Qwen3ASRPipelineBase(const std::filesystem::path& models_path, const ov::AnyMap& properties)
    : Qwen3ASRPipelineBase(models_path, Tokenizer{models_path}, properties) {}

Qwen3ASRPipelineBase::Qwen3ASRPipelineBase(const std::filesystem::path& models_path,
                                           const Tokenizer& tokenizer,
                                           const ov::AnyMap& properties)
    : ASRPipelineImplBase(models_path, tokenizer),
      m_feature_extractor{models_path / "preprocessor_config.json"},
      m_asr_text_token_id{get_required_token_id(m_tokenizer, "<asr_text>")},
      m_forced_aligner{get_forced_aligner(properties)} {
    // Qwen3-ASR EOS tokens: <|endoftext|>=151643, <|im_end|>=151645
    // The exported model has no generation_config.json. Qwen3-ASR original implementation hardcodes them as well.
    m_generation_config.set_eos_token_id(151643);
    m_generation_config.stop_token_ids.insert(151645);
}

ASRDecodedResults Qwen3ASRPipelineBase::generate(const AudiosView& audios,
                                                 const std::optional<ASRGenerationConfig>& generation_config,
                                                 const std::shared_ptr<StreamerBase> streamer) {
    get_single_audio(audios);
    const auto start_time = std::chrono::steady_clock::now();

    const ASRGenerationConfig config = resolve_generation_config(generation_config);

    ASRDecodedResults results;
    results.perf_metrics.load_time = m_load_time_ms;
    results.perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    const size_t max_chunk_sec = config.word_timestamps ? MAX_TIMESTAMPED_ASR_INPUT_SECONDS : MAX_ASR_INPUT_SECONDS;
    const std::vector<AudioChunk> chunks =
        split_audio_into_chunks(audios, m_feature_extractor.sampling_rate, max_chunk_sec);
    OPENVINO_ASSERT(!streamer || chunks.size() == 1,
                    "Qwen3-ASR streaming supports audio up to ",
                    max_chunk_sec,
                    " seconds. The input audio is split into ",
                    chunks.size(),
                    " chunks. Call generate() without a streamer.");

    const auto infer_results = infer(chunks, config, results.perf_metrics, streamer);
    if (streamer) {
        streamer->end();
    }

    const std::vector<std::pair<std::string, std::string>> parsed_results =
        parse_chunk_results(infer_results, config.language);

    auto [merged_texts, merged_languages] = merge_chunk_results(chunks, parsed_results);
    results.languages = std::move(merged_languages);
    for (std::string& text : merged_texts) {
        results.texts.push_back(std::move(text));
        results.scores.push_back(0.0f);
    }

    if (config.word_timestamps) {
        align_words(chunks, parsed_results, results);
    }

    const auto stop_time = std::chrono::steady_clock::now();
    results.perf_metrics.raw_metrics.generate_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(stop_time - start_time)));
    results.perf_metrics.evaluate_statistics(start_time);
    return results;
}

std::string Qwen3ASRPipelineBase::build_text_prompt(const ASRGenerationConfig& config) const {
    const std::string context = config.context.value_or("");

    // Hardcoded chat template to avoid tool-call warnings.
    std::string prompt = "<|im_start|>system\n" + context +
                         "<|im_end|>\n"
                         "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
                         "<|im_start|>assistant\n";

    if (config.language.has_value() && !config.language.value().empty()) {
        prompt += "language " + config.language.value() + "<asr_text>";
    }

    return prompt;
}

std::shared_ptr<StreamerBase> Qwen3ASRPipelineBase::create_chunk_streamer(const std::shared_ptr<StreamerBase>& streamer,
                                                                          const ASRGenerationConfig& config) const {
    if (!streamer) {
        return nullptr;
    }
    const bool forced_language = config.language.has_value() && !config.language.value().empty();
    return std::make_shared<Qwen3ASRStreamer>(streamer, m_asr_text_token_id, !forced_language);
}

std::pair<std::string, std::string> Qwen3ASRPipelineBase::parse_asr_output(
    const std::string& raw,
    const std::optional<std::string>& forced_language) const {
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

std::vector<std::pair<std::string, std::string>> Qwen3ASRPipelineBase::parse_chunk_results(
    const std::vector<std::string>& infer_results,
    const std::optional<std::string>& forced_language) const {
    std::vector<std::pair<std::string, std::string>> parsed_results;
    parsed_results.reserve(infer_results.size());
    for (const std::string& raw : infer_results) {
        parsed_results.push_back(parse_asr_output(raw, forced_language));
    }
    return parsed_results;
}

std::pair<std::vector<std::string>, std::vector<std::string>> Qwen3ASRPipelineBase::merge_chunk_results(
    const std::vector<AudioChunk>& chunks,
    const std::vector<std::pair<std::string, std::string>>& parsed_results) const {
    const size_t num_samples = get_num_samples(chunks);

    std::vector<std::string> merged_texts(num_samples);
    std::vector<std::string> merged_languages(num_samples);
    for (size_t i = 0; i < chunks.size(); ++i) {
        const size_t orig = chunks[i].orig_batch;
        const auto& [language, text] = parsed_results[i];
        if (!merged_texts[orig].empty()) {
            merged_texts[orig] += ' ';
        }
        merged_texts[orig] += text;
        merged_languages[orig] = language;  // Use the language reported for the last chunk.
    }

    return {std::move(merged_texts), std::move(merged_languages)};
}

void Qwen3ASRPipelineBase::align_words(const std::vector<AudioChunk>& chunks,
                                       const std::vector<std::pair<std::string, std::string>>& parsed_results,
                                       ASRDecodedResults& results) {
    std::vector<std::vector<ASRDecodedResultChunk>> words(get_num_samples(chunks));

    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& [language, text] = parsed_results[i];
        if (text.empty()) {
            continue;
        }

        const auto align_start = std::chrono::steady_clock::now();
        std::vector<ASRDecodedResultChunk> chunk_words =
            m_forced_aligner->align(chunks[i].wav, text, ov::genai::language(language));
        const auto align_stop = std::chrono::steady_clock::now();
        results.perf_metrics.asr_raw_metrics.word_level_timestamps_processing_durations.emplace_back(
            MicroSeconds(PerfMetrics::get_microsec(align_stop - align_start)));

        const float chunk_offset_sec = chunks[i].offset_sec;
        for (ASRDecodedResultChunk& word : chunk_words) {
            word.start_ts += chunk_offset_sec;
            word.end_ts += chunk_offset_sec;
        }

        if (!chunk_words.empty()) {
            std::vector<ASRDecodedResultChunk>& sample_words = words[chunks[i].orig_batch];
            sample_words.insert(sample_words.end(),
                                std::make_move_iterator(chunk_words.begin()),
                                std::make_move_iterator(chunk_words.end()));
        }
    }

    results.words = std::move(words);
}

ASRGenerationConfig Qwen3ASRPipelineBase::resolve_generation_config(
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

void Qwen3ASRPipelineBase::validate_generation_config(const ASRGenerationConfig& config) const {
    config.validate();

    OPENVINO_ASSERT(!config.is_beam_search(), "Qwen3-ASR does not support beam search decoding");

    OPENVINO_ASSERT(!config.word_timestamps || m_forced_aligner,
                    "Qwen3-ASR word timestamps require a forced aligner. Construct ASRPipeline with "
                    "ov::genai::forced_aligner(aligner) passing a pre-constructed ASRForcedAligner.");
}

}  // namespace ov::genai
