// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <vector>

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

    // Audio streamer extracted from AnyMap, forwarded to speech pipeline
    OmniSpeechStreamerVariant m_pending_speech_streamer = std::monostate{};

    // Speech config extracted from AnyMap: speech is decoded in the language model stream (LFM3-Omni)
    std::optional<OmniTalkerSpeechConfig> m_pending_speech_config;

    // RAII guard to reset the speech streamer and speech config on scope exit (exception-safe)
    struct SpeechPropertiesGuard {
        OmniSpeechStreamerVariant& streamer_ref;
        std::optional<OmniTalkerSpeechConfig>& speech_config_ref;
        ~SpeechPropertiesGuard() {
            streamer_ref = std::monostate{};
            speech_config_ref.reset();
        }
    };

    /// @brief Throws unless the language model emits speech frames itself, this backend decodes them, and the
    /// config fits the model.
    virtual void check_speech_in_stream(const OmniTalkerSpeechConfig& speech_config) const;

    /// @brief Reads talker_speech_config and speech_streamer from AnyMap into the pending members.
    void extract_speech_properties(const ov::AnyMap& config_map);

    GenerationConfig resolve_generation_config(const ov::AnyMap& config_map);

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
