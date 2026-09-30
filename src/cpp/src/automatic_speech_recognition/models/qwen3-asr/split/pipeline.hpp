// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "automatic_speech_recognition/models/qwen3-asr/audio_chunk.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"
#include "decoder.hpp"
#include "encoder.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

class Qwen3EncoderDecoderASR : public ASRPipelineImplBase {
public:
    Qwen3EncoderDecoderASR(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties);

    ASRDecodedResults generate(const AudioInputs& audio_inputs,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer = nullptr) override;

private:
    WhisperFeatureExtractor m_feature_extractor;
    const int64_t m_asr_text_token_id;

    static constexpr size_t MAX_ASR_INPUT_SECONDS = 1200;

    std::vector<std::string> infer(std::vector<AudioChunk> chunks,
                                   const ASRGenerationConfig& config,
                                   ASRPerfMetrics& perf_metrics,
                                   const std::shared_ptr<StreamerBase>& streamer_ptr = nullptr);

    ASRGenerationConfig resolve_generation_config(const std::optional<ASRGenerationConfig>& generation_config) const;

    void validate_generation_config(const ASRGenerationConfig& config) const;
};

}  // namespace ov::genai
