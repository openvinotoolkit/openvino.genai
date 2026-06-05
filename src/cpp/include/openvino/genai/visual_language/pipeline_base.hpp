// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

/**
 * @brief Internal Omni-aware base for `VLMPipeline` implementations.
 */
class OPENVINO_GENAI_EXPORTS VLMPipeline::VLMPipelineBase : public ov::genai::VLMPipelineBase {
    float m_load_time_ms = 0.0f;
    std::string m_attention_backend;

protected:
    VLMPipelineBase();

    // Audio tensors extracted from AnyMap, forwarded to implementations
    std::vector<ov::Tensor> m_pending_audios;
    // Audio streamer extracted from AnyMap, forwarded to speech pipeline
    OmniSpeechStreamerVariant m_pending_speech_streamer = std::monostate{};

    // RAII guard to reset audio streamer and pending audios on scope exit (exception-safe)
    struct AudioStreamerGuard {
        OmniSpeechStreamerVariant& streamer_ref;
        std::vector<ov::Tensor>& audios_ref;
        ~AudioStreamerGuard() {
            streamer_ref = std::monostate{};
            audios_ref.clear();
        }
    };

    GenerationConfig resolve_generation_config(const ov::AnyMap& config_map);

    static std::vector<ov::Tensor> extract_audios_from_config_map(const ov::AnyMap& config_map);

public:
    ~VLMPipelineBase() override;

    // Bring in the generate() overloads declared on the public abstract base so callers
    // that hold `VLMPipeline::VLMPipelineBase*` can reach all of them (otherwise the
    // metadata-aware overloads below would hide the inherited ones via name lookup rules).
    using ov::genai::VLMPipelineBase::generate;

    /// @brief Internal-only overload that takes `videos_metadata`. Public-base inheritance
    /// delivers the videos-with-metadata path through this entry point so OmniPipeline can
    /// drive it directly.
    virtual VLMDecodedResults generate(const std::string& prompt,
                                       const std::vector<ov::Tensor>& images,
                                       const std::vector<ov::Tensor>& videos,
                                       const std::vector<VideoMetadata>& videos_metadata,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) = 0;

    /// @brief Internal-only metadata-aware ChatHistory path.
    virtual VLMDecodedResults generate(const ChatHistory& history,
                                       const std::vector<ov::Tensor>& images,
                                       const std::vector<ov::Tensor>& videos,
                                       const std::vector<VideoMetadata>& videos_metadata,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) = 0;

    /// @brief Public-base AnyMap entry — extracts audios/streamer/vision props then delegates.
    VLMDecodedResults generate(const std::string& prompt, const ov::AnyMap& config_map) override;
    VLMDecodedResults generate(const ChatHistory& history, const ov::AnyMap& config_map) override;

    /// @brief Activate chat mode (legacy stateful path; deprecated on `VLMPipeline` itself
    /// but kept on the internal base because OmniPipeline / pipeline_base.cpp still calls
    /// these to thread through impls).
    virtual void start_chat(const std::string& system_message) = 0;
    virtual void finish_chat() = 0;

    /// @brief Enable or disable hidden-states accumulation in the underlying CB pipeline.
    /// OmniPipeline toggles this for the duration of an audio-producing generate() call
    /// via HiddenStatesCollectionScope. Default is off.
    virtual void enable_hidden_states_collection(bool enabled) = 0;

    /// @brief True when the backing pipeline implementation actually supports hidden-states
    /// collection (only the continuous-batching adapter does today). The SDPA fallback path
    /// returns false. OmniPipeline asserts on this at construction so users hit a clear
    /// "speech requires the CB backend" error instead of a cryptic late-stage failure
    /// inside run_with_speech.
    virtual bool supports_hidden_states_collection() const = 0;

    /// @brief True when the loaded VLM model has audio output enabled (Qwen3-Omni only).
    /// Used by OmniPipeline's ctor to reject non-Omni-capable models early.
    virtual bool is_audio_output_enabled() const = 0;

    /// @brief RAII helper that flips `enable_hidden_states_collection` on entry and off on destruction.
    /// Per-call scope keeps concurrent OmniPipeline instances independent — each owns its own VLM,
    /// so each scope only touches that VLM's flag.
    class OPENVINO_GENAI_EXPORTS HiddenStatesCollectionScope {
        VLMPipelineBase* m_pipeline;

    public:
        explicit HiddenStatesCollectionScope(VLMPipelineBase& pipeline);
        ~HiddenStatesCollectionScope();
        HiddenStatesCollectionScope(const HiddenStatesCollectionScope&) = delete;
        HiddenStatesCollectionScope& operator=(const HiddenStatesCollectionScope&) = delete;
        HiddenStatesCollectionScope(HiddenStatesCollectionScope&& other) noexcept;
        HiddenStatesCollectionScope& operator=(HiddenStatesCollectionScope&&) = delete;
    };

    /// @brief Set audio inputs to be forwarded to the underlying CB pipeline on the next generate() call.
    /// Internal-use API for OmniPipeline which constructs an OmniSpeechGenerationConfig path that
    /// bypasses the AnyMap audio extraction.
    void set_pending_audios(const std::vector<ov::Tensor>& audios) {
        m_pending_audios = audios;
    }

    /// @brief Clear any pending audio inputs. Internal-use API for OmniPipeline.
    void clear_pending_audios() {
        m_pending_audios.clear();
    }

    void set_attention_backend(const std::string& attention_backend) {
        m_attention_backend = attention_backend;
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
