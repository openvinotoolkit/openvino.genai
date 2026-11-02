// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "automatic_speech_recognition/models/qwen3-asr/pipeline_base.hpp"
#include "decoder.hpp"
#include "encoder.hpp"

namespace ov::genai {

class Qwen3EncoderDecoderASR : public Qwen3ASRPipelineBase {
public:
    Qwen3EncoderDecoderASR(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties);

private:
    std::unique_ptr<Qwen3ASREncoder> m_encoder;
    std::unique_ptr<Qwen3ASRDecoder> m_decoder;

    std::vector<std::string> infer(const std::vector<AudioChunk>& chunks,
                                   const ASRGenerationConfig& config,
                                   ASRPerfMetrics& perf_metrics,
                                   const std::shared_ptr<StreamerBase>& streamer) override;

    std::string extend_audio_tokens(const std::string& prompt, size_t audio_length) const;
};

}  // namespace ov::genai
