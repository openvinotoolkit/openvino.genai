// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "automatic_speech_recognition/pipeline_base.hpp"

namespace ov::genai {

struct VoxtralRealtimeRuntime;

class VoxtralASR final : public ASRPipelineImplBase {
public:
    VoxtralASR(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    ASRDecodedResults generate(const AudioInputs& audio_inputs,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer) override;

    std::unique_ptr<ASRStreamImplBase> create_stream(
        const std::optional<ASRGenerationConfig>& generation_config) override;

private:
    std::shared_ptr<VoxtralRealtimeRuntime> m_runtime;
};

}  // namespace ov::genai