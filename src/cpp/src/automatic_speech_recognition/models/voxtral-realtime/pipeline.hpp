// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "automatic_speech_recognition/pipeline_base.hpp"

namespace ov::genai {

struct VoxtralRealtimeRuntime;

/// Layout of an exported Voxtral Realtime model.
enum class VoxtralExport {
    /// openvino_audio_encoder.xml and openvino_text_decoder.xml. Text decoder embeds tokens itself and takes its KV
    /// cache as inputs.
    ENCODER_DECODER,
    /// audio_encoder, text_embeddings and text_decoder directories. Stateful text decoder takes input embeddings and
    /// runs on PagedAttention backend.
    SPLIT,
};

class VoxtralASR final : public ASRPipelineImplBase {
public:
    VoxtralASR(const std::filesystem::path& models_path,
               const std::string& device,
               const ov::AnyMap& properties,
               VoxtralExport model_export = VoxtralExport::ENCODER_DECODER);

    ASRDecodedResults generate(const AudiosView& audios,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer) override;

    std::unique_ptr<ASRStreamImplBase> create_stream(
        const std::optional<ASRGenerationConfig>& generation_config) override;

    std::unique_ptr<ASRStreamingSessionImpl> create_streaming_session_impl(
        const ASRStreamingConfig& streaming_config,
        const ASRGenerationConfig& generation_config) override;

    /// Every session owns a separate audio encoder infer request. Text decoder infer request is owned by the session
    /// too for encoder-decoder export, and is shared by the sessions for split export, which batches their inference.
    size_t max_streaming_sessions() const override {
        return std::numeric_limits<size_t>::max();
    }

private:
    std::shared_ptr<VoxtralRealtimeRuntime> m_runtime;
};

/**
 * @brief Create Voxtral Realtime split export pipeline. Its text decoder runs on PagedAttention backend, which batches
 * decoding steps of concurrent streaming sessions and generate() calls. SDPA backend isn't supported.
 */
std::unique_ptr<ASRPipelineImplBase> create_voxtral_split_pipeline(const std::filesystem::path& models_path,
                                                                   const std::string& device,
                                                                   const ov::AnyMap& properties);

}  // namespace ov::genai
