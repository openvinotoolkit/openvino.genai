// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/streamer_base.hpp"

#include "whisper_pipeline_base.hpp"
#include "whisper/whisper_models.hpp"

namespace ov {
namespace genai {

class WhisperPipeline::StaticWhisperPipeline : public WhisperPipeline::WhisperPipelineImplBase {
public:
    StaticWhisperPipeline(const std::filesystem::path& model_path,
                          const ov::genai::Tokenizer& tokenizer,
                          const ov::AnyMap& plugin_config);

    StaticWhisperPipeline(const std::filesystem::path& model_path,
                          const ov::AnyMap& plugin_config);

    DecodedResults generate(const RawSpeechInput& raw_speech_input,
                            OptionalWhisperGenerationConfig generation_config,
                            StreamerVariant streamer) override;
private:
    WhisperInitializedModels m_models;
};

}  // namespace genai
}  // namespace ov
