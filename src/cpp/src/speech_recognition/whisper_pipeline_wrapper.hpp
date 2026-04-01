// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "asr_pipeline_impl_base.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {

/**
 * @brief ASR back-end wrapper for Whisper models.
 *
 * Directly delegates to ov::genai::WhisperPipeline which already
 * handles stateful / static variants, tokenizer, feature extraction,
 * sampling, timestamps, streamers, perf-metrics, etc.
 * 
 * This is a thin wrapper that adapts WhisperPipeline to the ASRPipelineImplBase
 * interface, enabling unified ASR pipeline usage.
 */
class WhisperPipelineWrapper : public ASRPipelineImplBase {
public:
    WhisperPipelineWrapper(const std::filesystem::path& model_dir,
                           const std::string& device,
                           const ov::AnyMap& properties)
        : ASRPipelineImplBase(model_dir, device, properties)
        , m_whisper(model_dir, device, properties) {}

    ~WhisperPipelineWrapper() override = default;

    // ── Core inference (Generic API) ────────────────────────────────────

    ASRDecodedResults generate(
        const RawSpeechInput& raw_speech_input,
        const ASRGenerationConfig& config,
        const std::shared_ptr<StreamerBase> streamer) override {
        
        // Convert ASRGenerationConfig to WhisperGenerationConfig
        WhisperGenerationConfig whisper_config = m_whisper.get_generation_config();
        whisper_config.max_new_tokens = config.max_new_tokens;
        
        // Handle language field: empty string means unset (auto-detect)
        if (config.language.empty()) {
            whisper_config.language = std::nullopt;
        } else {
            whisper_config.language = config.language;
        }
        
        // Handle task field: empty string means unset (use default)
        if (config.task.empty()) {
            whisper_config.task = std::nullopt;
        } else {
            whisper_config.task = config.task;
        }
        
        whisper_config.return_timestamps = config.return_timestamps;
        whisper_config.temperature = config.temperature;
        
        // Map additional decoding parameters that exist in GenerationConfig base class
        // Note: WhisperGenerationConfig inherits these from GenerationConfig
        if (config.top_k > 0) {
            whisper_config.top_k = config.top_k;
        }
        if (config.top_p < 1.0f) {
            whisper_config.top_p = config.top_p;
        }
        if (config.num_beams > 1) {
            whisper_config.num_beams = config.num_beams;
        }
        
        // Build StreamerVariant from shared_ptr<StreamerBase>
        StreamerVariant sv = streamer ? StreamerVariant{streamer} : StreamerVariant{std::monostate{}};
        
        // Call WhisperPipeline
        WhisperDecodedResults whisper_results = m_whisper.generate(raw_speech_input, whisper_config, sv);
        
        // Convert to generic ASRDecodedResults
        ASRDecodedResults results;
        results.texts = whisper_results.texts;
        results.scores = whisper_results.scores;
        results.perf_metrics = whisper_results.perf_metrics;
        
        // Convert timestamps if present
        if (whisper_results.chunks.has_value() && !whisper_results.chunks->empty()) {
            std::vector<std::vector<ASRTimestampChunk>> all_timestamps;
            std::vector<ASRTimestampChunk> ts_chunks;
            for (const auto& chunk : *whisper_results.chunks) {
                ASRTimestampChunk ts;
                ts.start_ts = chunk.start_ts;
                ts.end_ts = chunk.end_ts;
                ts.text = chunk.text;
                ts_chunks.push_back(ts);
            }
            all_timestamps.push_back(ts_chunks);
            results.timestamps = all_timestamps;
        }
        
        return results;
    }

    // ── Legacy Whisper-compatible API ───────────────────────────────────

    WhisperDecodedResults generate_whisper(
        const RawSpeechInput& raw_speech_input,
        OptionalWhisperGenerationConfig generation_config,
        const std::shared_ptr<StreamerBase> streamer) override {
        
        StreamerVariant sv = streamer ? StreamerVariant{streamer} : StreamerVariant{std::monostate{}};
        return m_whisper.generate(raw_speech_input, generation_config, sv);
    }

    // ── Accessors ───────────────────────────────────────────────────────

    Tokenizer get_tokenizer() override {
        return m_whisper.get_tokenizer();
    }

    ASRGenerationConfig get_generation_config() const override {
        WhisperGenerationConfig wc = m_whisper.get_generation_config();
        ASRGenerationConfig config;
        config.max_new_tokens = wc.max_new_tokens;
        config.language = wc.language.value_or("");
        config.task = wc.task.value_or("");
        config.return_timestamps = wc.return_timestamps;
        config.temperature = wc.temperature;
        // These come from GenerationConfig base class
        config.top_k = wc.top_k;
        config.top_p = wc.top_p;
        config.num_beams = wc.num_beams;
        return config;
    }

    void set_generation_config(const ASRGenerationConfig& config) override {
        WhisperGenerationConfig wc = m_whisper.get_generation_config();
        wc.max_new_tokens = config.max_new_tokens;
        
        // Empty string means unset/auto-detect
        if (config.language.empty()) {
            wc.language = std::nullopt;
        } else {
            wc.language = config.language;
        }
        
        if (config.task.empty()) {
            wc.task = std::nullopt;
        } else {
            wc.task = config.task;
        }
        
        wc.return_timestamps = config.return_timestamps;
        wc.temperature = config.temperature;
        
        // These come from GenerationConfig base class
        if (config.top_k > 0) {
            wc.top_k = config.top_k;
        }
        if (config.top_p < 1.0f) {
            wc.top_p = config.top_p;
        }
        if (config.num_beams > 1) {
            wc.num_beams = config.num_beams;
        }
        
        m_whisper.set_generation_config(wc);
    }

    WhisperGenerationConfig get_whisper_generation_config() const override {
        return m_whisper.get_generation_config();
    }

    void set_whisper_generation_config(const WhisperGenerationConfig& config) override {
        m_whisper.set_generation_config(config);
    }

private:
    WhisperPipeline m_whisper;
};

}  // namespace genai
}  // namespace ov
