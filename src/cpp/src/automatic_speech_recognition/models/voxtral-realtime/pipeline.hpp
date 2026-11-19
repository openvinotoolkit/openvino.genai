// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "automatic_speech_recognition/pipeline_base.hpp"

namespace ov::genai {

struct VoxtralRealtimeRuntime;

class VoxtralASR final : public ASRPipelineImplBase {
public:
    VoxtralASR(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    ASRDecodedResults generate(const AudiosView& audios,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer) override;

    std::unique_ptr<ASRStreamImplBase> create_stream(
        const std::optional<ASRGenerationConfig>& generation_config) override;

    std::unique_ptr<ASRStreamingSessionImpl> create_streaming_session_impl(
        const ASRStreamingConfig& streaming_config,
        const ASRGenerationConfig& generation_config) override;

    /// Every session owns separate audio encoder and text decoder infer requests.
    size_t max_streaming_sessions() const override {
        return std::numeric_limits<size_t>::max();
    }

private:
    std::shared_ptr<VoxtralRealtimeRuntime> m_runtime;
};

}  // namespace ov::genai
