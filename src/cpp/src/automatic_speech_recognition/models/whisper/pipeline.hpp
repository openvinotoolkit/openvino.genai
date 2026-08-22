// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>

#include "automatic_speech_recognition/pipeline_base.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "utils.hpp"

namespace ov::genai {

class WhisperASRPipelineAdapter : public ASRPipelineImplBase {
public:
    WhisperASRPipelineAdapter(const std::filesystem::path& models_path,
                              const std::string& device,
                              const ov::AnyMap& properties)
        : ASRPipelineImplBase(models_path),
          m_whisper_pipeline(models_path, device, properties) {}

    void set_generation_config(const ASRGenerationConfig& config) override {
        ASRPipelineImplBase::set_generation_config(config);
        m_whisper_pipeline.set_generation_config(to_whisper_config(config));
    }

    ASRDecodedResults generate(const AudioInputs& audio_inputs,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer) override {
        OptionalWhisperGenerationConfig whisper_config = std::nullopt;
        if (generation_config.has_value()) {
            whisper_config = to_whisper_config(generation_config.value());
        }

        StreamerVariant streamer_variant = streamer ? StreamerVariant{streamer} : StreamerVariant{std::monostate{}};

        std::vector<WhisperDecodedResults> whisper_results = std::visit(
            ov::genai::utils::overloaded{
                [&](const std::vector<float>& input) -> std::vector<WhisperDecodedResults> {
                    std::vector<WhisperDecodedResults> single;
                    single.push_back(m_whisper_pipeline.generate(input, std::move(whisper_config), streamer_variant));
                    return single;
                },
                [&](const std::vector<std::vector<float>>& inputs) -> std::vector<WhisperDecodedResults> {
                    return m_whisper_pipeline.generate(inputs, std::move(whisper_config), streamer_variant);
                },
            },
            audio_inputs);

        return to_asr_results(std::move(whisper_results));
    }

private:
    WhisperPipeline m_whisper_pipeline;

    static WhisperGenerationConfig to_whisper_config(const ASRGenerationConfig& asr_config) {
        WhisperGenerationConfig config;
        static_cast<GenerationConfig&>(config) = static_cast<const GenerationConfig&>(asr_config);

        config.decoder_start_token_id = asr_config.decoder_start_token_id;
        config.pad_token_id = asr_config.pad_token_id;
        config.translate_token_id = asr_config.translate_token_id;
        config.transcribe_token_id = asr_config.transcribe_token_id;
        config.prev_sot_token_id = asr_config.prev_sot_token_id;
        config.no_timestamps_token_id = asr_config.no_timestamps_token_id;
        config.max_initial_timestamp_index = asr_config.max_initial_timestamp_index;
        config.is_multilingual = asr_config.is_multilingual;
        config.language = asr_config.language;
        config.lang_to_id = asr_config.lang_to_id;
        config.task = asr_config.task;
        config.return_timestamps = asr_config.return_timestamps;
        config.word_timestamps = asr_config.word_timestamps;
        config.alignment_heads = asr_config.alignment_heads;
        config.initial_prompt = asr_config.initial_prompt;
        config.hotwords = asr_config.hotwords;
        config.begin_suppress_tokens = asr_config.begin_suppress_tokens;
        config.suppress_tokens = asr_config.suppress_tokens;

        return config;
    }

    static ASRDecodedResults to_asr_results(std::vector<WhisperDecodedResults>&& whisper_results) {
        OPENVINO_ASSERT(!whisper_results.empty(), "Internal error: Whisper produced no result for the ASR pipeline.");

        ASRDecodedResults result;
        result.texts.reserve(whisper_results.size());
        result.scores.reserve(whisper_results.size());
        result.languages.reserve(whisper_results.size());

        // Preserve input order across all result fields.
        for (auto& whisper_result : whisper_results) {
            OPENVINO_ASSERT(whisper_result.texts.size() == 1 && whisper_result.scores.size() == 1,
                            "Internal error: expected exactly one Whisper sequence per audio.");
            result.texts.push_back(std::move(whisper_result.texts.front()));
            result.scores.push_back(whisper_result.scores.front());
            result.languages.push_back(std::move(whisper_result.language));
        }

        // Keep chunk and word indices aligned with the input batch.
        const bool has_chunks = std::any_of(whisper_results.begin(), whisper_results.end(), [](const auto& item) {
            return item.chunks.has_value();
        });
        if (has_chunks) {
            std::vector<std::vector<ASRDecodedResultChunk>> all_chunks;
            all_chunks.reserve(whisper_results.size());
            for (auto& whisper_result : whisper_results) {
                std::vector<ASRDecodedResultChunk> chunks;
                if (whisper_result.chunks.has_value()) {
                    chunks.reserve(whisper_result.chunks->size());
                    for (auto& chunk : *whisper_result.chunks) {
                        chunks.push_back({chunk.start_ts, chunk.end_ts, std::move(chunk.text), {}});
                    }
                }
                all_chunks.push_back(std::move(chunks));
            }
            result.chunks = std::move(all_chunks);
        }

        const bool has_words = std::any_of(whisper_results.begin(), whisper_results.end(), [](const auto& item) {
            return item.words.has_value();
        });
        if (has_words) {
            std::vector<std::vector<ASRDecodedResultChunk>> all_words;
            all_words.reserve(whisper_results.size());
            for (auto& whisper_result : whisper_results) {
                std::vector<ASRDecodedResultChunk> words;
                if (whisper_result.words.has_value()) {
                    words.reserve(whisper_result.words->size());
                    for (auto& word : *whisper_result.words) {
                        words.push_back({word.start_ts, word.end_ts, std::move(word.word), std::move(word.token_ids)});
                    }
                }
                all_words.push_back(std::move(words));
            }
            result.words = std::move(all_words);
        }

        // Batch results share aggregate metrics; convert them once.
        WhisperPerfMetrics& aggregate_metrics = whisper_results.front().perf_metrics;
        PerfMetrics base_metrics = static_cast<PerfMetrics&>(aggregate_metrics);
        result.perf_metrics = ASRPerfMetrics{base_metrics};
        result.perf_metrics.asr_raw_metrics.features_extraction_durations =
            std::move(aggregate_metrics.whisper_raw_metrics.features_extraction_durations);
        result.perf_metrics.asr_raw_metrics.word_level_timestamps_processing_durations =
            std::move(aggregate_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations);
        result.perf_metrics.asr_raw_metrics.encode_inference_durations =
            std::move(aggregate_metrics.whisper_raw_metrics.encode_inference_durations);
        result.perf_metrics.asr_raw_metrics.decode_inference_durations =
            std::move(aggregate_metrics.whisper_raw_metrics.decode_inference_durations);
        result.perf_metrics.features_extraction_duration = aggregate_metrics.features_extraction_duration;
        result.perf_metrics.word_level_timestamps_processing_duration =
            aggregate_metrics.word_level_timestamps_processing_duration;
        result.perf_metrics.encode_inference_duration = aggregate_metrics.encode_inference_duration;
        result.perf_metrics.decode_inference_duration = aggregate_metrics.decode_inference_duration;

        return result;
    }
};

}  // namespace ov::genai
