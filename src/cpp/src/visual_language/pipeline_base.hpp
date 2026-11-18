// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "omni/speech_streamer_utils.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

/**
 * @brief Internal Omni-aware base for `VLMPipeline` implementations.
 */
class VLMPipeline::VLMBackend : public ov::genai::VLMPipelineBase {
    float m_load_time_ms = 0.0f;
    std::string m_attention_backend;

protected:
    VLMBackend();

    /// @brief Speech output requested in the AnyMap generate(), decoded in the language model stream (LFM3-Omni).
    struct SpeechRequest {
        OmniTalkerSpeechConfig config;
        OmniSpeechStreamerVariant streamer;
    };

    /// @brief Throws unless the language model emits speech frames itself, this backend decodes them, and the
    /// config fits the model.
    virtual void check_speech_in_stream(const OmniTalkerSpeechConfig& speech_config) const;

    /// @brief The typed generate() with the speech request of the AnyMap generate(). The default ignores the request:
    /// check_speech_in_stream() has already rejected speech for backends that cannot decode it.
    virtual VLMDecodedResults generate_with_speech(const std::string& prompt,
                                                   const std::vector<ov::Tensor>& images,
                                                   const std::vector<ov::Tensor>& videos,
                                                   const std::vector<ov::Tensor>& audios,
                                                   const std::vector<VideoMetadata>& videos_metadata,
                                                   const GenerationConfig& generation_config,
                                                   const StreamerVariant& streamer,
                                                   const std::optional<SpeechRequest>& speech_request);
    virtual VLMDecodedResults generate_with_speech(const ChatHistory& history,
                                                   const std::vector<ov::Tensor>& images,
                                                   const std::vector<ov::Tensor>& videos,
                                                   const std::vector<ov::Tensor>& audios,
                                                   const std::vector<VideoMetadata>& videos_metadata,
                                                   const GenerationConfig& generation_config,
                                                   const StreamerVariant& streamer,
                                                   const std::optional<SpeechRequest>& speech_request);

    GenerationConfig resolve_generation_config(const ov::AnyMap& config_map);

private:
    /// @brief Reads talker_speech_config and, when it requests audio, speech_streamer from AnyMap. A request for audio
    /// is validated by check_speech_request(), so the caller already guards the streamer when the config is rejected.
    static std::optional<SpeechRequest> read_speech_request(const ov::AnyMap& config_map);
    void check_speech_request(const SpeechRequest& speech_request) const;

public:
    ~VLMBackend() override;

    // Bring in the generate() overloads declared on the public abstract base so callers
    // that hold `VLMPipeline::VLMBackend*` can reach all of them (otherwise the
    // AnyMap overrides below would hide the inherited ones via name lookup rules).
    // The videos_metadata-aware overloads and the Omni hooks are declared on VLMPipelineBase.
    using ov::genai::VLMPipelineBase::generate;

    /// @brief Public-base AnyMap entry — extracts audios/streamer/vision props then delegates.
    VLMDecodedResults generate(const std::string& prompt, const ov::AnyMap& config_map) override;
    VLMDecodedResults generate(const ChatHistory& history, const ov::AnyMap& config_map) override;

    /// @brief Activate chat mode (legacy stateful path; deprecated on `VLMPipeline` itself
    /// but kept on the internal base because OmniPipeline / pipeline_base.cpp still calls
    /// these to thread through impls).
    virtual void start_chat(const std::string& system_message) = 0;
    virtual void finish_chat() = 0;

    void set_attention_backend(const std::string& attention_backend) {
        m_attention_backend = attention_backend;
    }

    const std::string& get_attention_backend() const {
        return m_attention_backend;
    }

    void set_load_time(float load_time_ms) {
        m_load_time_ms = load_time_ms;
    }

    float get_load_time() const {
        return m_load_time_ms;
    }
};

}  // namespace genai
}  // namespace ov
