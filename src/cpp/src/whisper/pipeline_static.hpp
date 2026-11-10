// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "whisper/models.hpp"
#include "whisper/pipeline_base.hpp"
#include "sampling/sampler.hpp"

namespace ov {
namespace genai {

class StaticWhisperPipeline : public WhisperPipelineImplBase {
public:
    StaticWhisperPipeline(const std::filesystem::path& model_path, const ov::AnyMap& properties);

    std::vector<WhisperDecodedResults> generate(const AudiosView& audios,
                                                OptionalWhisperGenerationConfig generation_config,
                                                const std::shared_ptr<StreamerBase> streamer) override;

private:
    WhisperDecodedResults generate_single(const RawSpeechInput& raw_speech_input,
                                          OptionalWhisperGenerationConfig generation_config,
                                          const std::shared_ptr<StreamerBase> streamer);

    WhisperInitializedModels m_models;
    Sampler m_sampler;
};

}  // namespace genai
}  // namespace ov
