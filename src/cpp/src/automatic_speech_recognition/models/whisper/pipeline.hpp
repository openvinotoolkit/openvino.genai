// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "automatic_speech_recognition/pipeline_base.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "utils.hpp"
#include "whisper/pipeline_base.hpp"

namespace ov::genai {

class WhisperASRPipelineAdapter : public ASRPipelineImplBase {
public:
    WhisperASRPipelineAdapter(const std::filesystem::path& models_path,
                              const std::string& device,
                              const ov::AnyMap& properties)
        : WhisperASRPipelineAdapter(models_path, create_whisper_pipeline_impl(models_path, device, properties)) {}

    void set_generation_config(const ASRGenerationConfig& incoming_config) override {
        ASRGenerationConfig config = incoming_config;
        config.validate();
        m_whisper->set_generation_config(to_whisper_config(config));
        m_generation_config = std::move(config);
    }

    ASRDecodedResults generate(const AudiosView& audios,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer) override {
        OptionalWhisperGenerationConfig whisper_config = std::nullopt;
        if (generation_config.has_value()) {
            whisper_config = to_whisper_config(generation_config.value());
        }

        ASRDecodedResults result = to_asr_results(m_whisper->generate(audios, std::move(whisper_config), streamer));
        result.perf_metrics.load_time = m_load_time_ms;
        return result;
    }

private:
    WhisperASRPipelineAdapter(const std::filesystem::path& models_path,
                              std::unique_ptr<WhisperPipelineImplBase> whisper)
        : ASRPipelineImplBase(models_path, whisper->m_tokenizer),
          m_whisper(std::move(whisper)) {}

    std::unique_ptr<WhisperPipelineImplBase> m_whisper;

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

    // Every Whisper result of one batched call carries the same metrics for the whole call.
    static ASRDecodedResults to_asr_results(std::vector<WhisperDecodedResults>&& whisper_results) {
        ASRDecodedResults result;
        for (size_t i = 0; i < whisper_results.size(); ++i) {
            ASRDecodedResults single = to_asr_results(std::move(whisper_results[i]));
            if (i == 0) {
                result.perf_metrics = std::move(single.perf_metrics);
            }
            result.texts.push_back(std::move(single.texts.front()));
            result.scores.push_back(single.scores.front());
            result.languages.push_back(std::move(single.languages.front()));
            if (single.chunks.has_value()) {
                if (!result.chunks.has_value()) {
                    result.chunks.emplace();
                }
                result.chunks->push_back(std::move(single.chunks->front()));
            }
            if (single.words.has_value()) {
                if (!result.words.has_value()) {
                    result.words.emplace();
                }
                result.words->push_back(std::move(single.words->front()));
            }
        }
        return result;
    }

    static ASRDecodedResults to_asr_results(WhisperDecodedResults&& whisper_result) {
        PerfMetrics base_metrics = static_cast<PerfMetrics&>(whisper_result.perf_metrics);
        ASRDecodedResults result;
        result.texts = std::move(whisper_result.texts);
        result.scores = std::move(whisper_result.scores);
        result.languages = std::vector<std::string>{std::move(whisper_result.language)};
        result.perf_metrics = ASRPerfMetrics{base_metrics};
        result.perf_metrics.asr_raw_metrics.features_extraction_durations =
            std::move(whisper_result.perf_metrics.whisper_raw_metrics.features_extraction_durations);
        result.perf_metrics.asr_raw_metrics.word_level_timestamps_processing_durations =
            std::move(whisper_result.perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations);
        result.perf_metrics.asr_raw_metrics.encode_inference_durations =
            std::move(whisper_result.perf_metrics.whisper_raw_metrics.encode_inference_durations);
        result.perf_metrics.asr_raw_metrics.decode_inference_durations =
            std::move(whisper_result.perf_metrics.whisper_raw_metrics.decode_inference_durations);
        result.perf_metrics.features_extraction_duration = whisper_result.perf_metrics.features_extraction_duration;
        result.perf_metrics.word_level_timestamps_processing_duration =
            whisper_result.perf_metrics.word_level_timestamps_processing_duration;
        result.perf_metrics.encode_inference_duration = whisper_result.perf_metrics.encode_inference_duration;
        result.perf_metrics.decode_inference_duration = whisper_result.perf_metrics.decode_inference_duration;

        if (whisper_result.chunks.has_value()) {
            std::vector<ASRDecodedResultChunk> chunks;
            chunks.reserve(whisper_result.chunks->size());
            for (auto& chunk : *whisper_result.chunks) {
                chunks.push_back({chunk.start_ts, chunk.end_ts, std::move(chunk.text), {}});
            }
            result.chunks = {std::move(chunks)};
        }

        if (whisper_result.words.has_value()) {
            std::vector<ASRDecodedResultChunk> words;
            words.reserve(whisper_result.words->size());
            for (auto& word : *whisper_result.words) {
                words.push_back({word.start_ts, word.end_ts, std::move(word.word), std::move(word.token_ids)});
            }
            result.words = {std::move(words)};
        }

        return result;
    }
};

}  // namespace ov::genai
