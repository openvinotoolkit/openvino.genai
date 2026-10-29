// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "automatic_speech_recognition/models/qwen3-asr/audio_chunk.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"
#include "openvino/genai/automatic_speech_recognition/forced_aligner.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

/**
 * @brief Qwen3-ASR flow shared by all model export layouts: audio chunking, prompt building, output parsing,
 * chunk merging and word-level alignment. Derived classes implement chunk transcription.
 */
class Qwen3ASRPipelineBase : public ASRPipelineImplBase {
public:
    ASRDecodedResults generate(const AudioInputs& audio_inputs,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer = nullptr) override;

protected:
    Qwen3ASRPipelineBase(const std::filesystem::path& models_path, const ov::AnyMap& properties);
    Qwen3ASRPipelineBase(const std::filesystem::path& models_path,
                         const Tokenizer& tokenizer,
                         const ov::AnyMap& properties);

    /**
     * @brief Transcribe audio chunks. A streamer is passed only for a single chunk.
     * @return Raw decoded text for each chunk, in the order of chunks.
     */
    virtual std::vector<std::string> infer(const std::vector<AudioChunk>& chunks,
                                           const ASRGenerationConfig& config,
                                           ASRPerfMetrics& perf_metrics,
                                           const std::shared_ptr<StreamerBase>& streamer) = 0;

    std::string build_text_prompt(const ASRGenerationConfig& config) const;

    /**
     * @brief Wrap a user streamer to suppress the language autodetection prefix of a single chunk.
     * @return nullptr when streamer is nullptr.
     */
    std::shared_ptr<StreamerBase> create_chunk_streamer(const std::shared_ptr<StreamerBase>& streamer,
                                                        const ASRGenerationConfig& config) const;

    WhisperFeatureExtractor m_feature_extractor;

private:
    const int64_t m_asr_text_token_id;
    std::shared_ptr<ASRForcedAligner> m_forced_aligner;

    static constexpr size_t MAX_TIMESTAMPED_ASR_INPUT_SECONDS = 180;

    std::pair<std::string, std::string> parse_asr_output(const std::string& raw,
                                                         const std::optional<std::string>& forced_language) const;
    std::vector<std::pair<std::string, std::string>> parse_chunk_results(
        const std::vector<std::string>& infer_results,
        const std::optional<std::string>& forced_language) const;
    std::pair<std::vector<std::string>, std::vector<std::string>> merge_chunk_results(
        const std::vector<AudioChunk>& chunks,
        const std::vector<std::pair<std::string, std::string>>& parsed_results) const;

    void align_words(const std::vector<AudioChunk>& chunks,
                     const std::vector<std::pair<std::string, std::string>>& parsed_results,
                     ASRDecodedResults& results);

    ASRGenerationConfig resolve_generation_config(const std::optional<ASRGenerationConfig>& generation_config) const;

    void validate_generation_config(const ASRGenerationConfig& config) const;
};

}  // namespace ov::genai
