// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/omni_pipeline.hpp"
#include "openvino/genai/omni_speech_generation_config.hpp"
#include "openvino/genai/omni_speech_streamer_base.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/pipeline_base.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/qwen3_omni/speech_pipeline.hpp"

namespace ov::genai {

/// @brief Implementation backing OmniPipeline. Holds a VLMPipelineBase (text-side) and a
/// Qwen3OmniSpeechPipeline (speech-side). Speech is gated per-call via
/// OmniSpeechGenerationConfig::return_audio.
class OmniPipeline::OmniPipelineImpl {
public:
    OmniPipelineImpl(const std::filesystem::path& models_path,
                     const std::string& device,
                     const ov::AnyMap& properties);

    OmniPipelineImpl(const std::shared_ptr<VLMPipeline::VLMPipelineBase>& vlm,
                     const std::filesystem::path& speech_models_path,
                     const std::string& device,
                     const ov::AnyMap& properties);

    VLMDecodedResults generate(const std::string& prompt,
                               const std::vector<ov::Tensor>& images,
                               const std::vector<ov::Tensor>& videos,
                               const std::vector<ov::Tensor>& audios,
                               const OmniSpeechGenerationConfig& speech_config,
                               const StreamerVariant& text_streamer,
                               const OmniSpeechStreamerVariant& speech_streamer);

    VLMDecodedResults generate(const ChatHistory& history,
                               const std::vector<ov::Tensor>& images,
                               const std::vector<ov::Tensor>& videos,
                               const std::vector<ov::Tensor>& audios,
                               const OmniSpeechGenerationConfig& speech_config,
                               const StreamerVariant& text_streamer,
                               const OmniSpeechStreamerVariant& speech_streamer);

    void start_chat(const std::string& system_message);

    void finish_chat();

private:
    /// @brief Reject non-Omni-capable models early, before any inference work.
    void assert_omni_capable() const;

    /// @brief Run the speech step against the hidden states collected during VLM generation.
    VLMDecodedResults run_with_speech(VLMDecodedResults vlm_result,
                                      const OmniSpeechGenerationConfig& speech_config,
                                      const OmniSpeechStreamerVariant& speech_streamer);

    /// @brief RAII guard that clears the shared VLM's pending-audios slot when the scope exits,
    /// even on exception. Keeps a stale audio batch from leaking into the next generate() call
    /// on the same VLM instance — important when the VLM is shared via the shared-VLM ctor.
    class PendingAudiosGuard {
    public:
        PendingAudiosGuard(VLMPipeline::VLMPipelineBase& vlm, const std::vector<ov::Tensor>& audios) : m_vlm(vlm) {
            m_vlm.set_pending_audios(audios);
        }
        ~PendingAudiosGuard() {
            m_vlm.clear_pending_audios();
        }
        PendingAudiosGuard(const PendingAudiosGuard&) = delete;
        PendingAudiosGuard& operator=(const PendingAudiosGuard&) = delete;

    private:
        VLMPipeline::VLMPipelineBase& m_vlm;
    };

    std::shared_ptr<VLMPipeline::VLMPipelineBase> m_vlm;
    std::unique_ptr<Qwen3OmniSpeechPipeline> m_speech;
};

}  // namespace ov::genai
