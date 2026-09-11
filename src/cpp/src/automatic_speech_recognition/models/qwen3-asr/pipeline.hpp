// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "audio_chunk.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"
#include "decoder.hpp"
#include "encoder.hpp"
#include "qwen3_forced_aligner.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

class Qwen3ASR : public ASRPipelineImplBase {
public:
    Qwen3ASR(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    ASRDecodedResults generate(const AudioInputs& audio_inputs,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer = nullptr) override;

private:
    WhisperFeatureExtractor m_feature_extractor;
    const int64_t m_asr_text_token_id;
    std::unique_ptr<Qwen3ASREncoder> m_encoder;
    std::unique_ptr<Qwen3ASRDecoder> m_decoder;
    std::unique_ptr<Qwen3ForcedAligner> m_forced_aligner;

    static constexpr size_t MAX_ASR_INPUT_SECONDS = 1200;
    static constexpr size_t MAX_TIMESTAMPED_ASR_INPUT_SECONDS = 180;

    std::vector<std::string> infer(std::vector<AudioChunk> chunks,
                                   const ASRGenerationConfig& config,
                                   ASRPerfMetrics& perf_metrics,
                                   const std::shared_ptr<StreamerBase>& streamer_ptr = nullptr);

    std::vector<std::string> build_text_prompt(size_t batch_size, const ASRGenerationConfig& config);
    std::pair<std::string, std::string> parse_asr_output(const std::string& raw,
                                                         const std::optional<std::string>& forced_language);
    std::vector<std::pair<std::string, std::string>> parse_chunk_results(
        const std::vector<std::string>& infer_results,
        const std::optional<std::string>& forced_language);
    std::pair<std::vector<std::string>, std::vector<std::string>> merge_chunk_results(
        const std::vector<AudioChunk>& chunks,
        const std::vector<std::pair<std::string, std::string>>& parsed_results);
    std::vector<std::string> extend_audio_tokens(const std::vector<std::string>& prompts,
                                                 const std::vector<size_t>& audio_lengths);

    void align_words(const std::vector<AudioChunk>& chunks,
                     const std::vector<std::pair<std::string, std::string>>& parsed_results,
                     ASRDecodedResults& results);

    ASRGenerationConfig resolve_generation_config(const std::optional<ASRGenerationConfig>& generation_config) const;

    void validate_generation_config(const ASRGenerationConfig& config) const;
};

}  // namespace ov::genai
