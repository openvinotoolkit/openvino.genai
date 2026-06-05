// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/omni/speech_generation_config.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/pipeline_base.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/// @brief Implementation backing OmniPipeline. Holds a VLMPipelineBase (text-side) and a
/// TalkerBase (speech-side). Speech is gated per-call via OmniSpeechGenerationConfig::return_audio.
class OmniPipeline::OmniPipelineImpl {
public:
    OmniPipelineImpl(const std::shared_ptr<VLMPipeline::VLMPipelineBase>& vlm,
                     const std::shared_ptr<TalkerBase>& talker);

    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<ov::Tensor>& audios,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer,
                                const OmniSpeechStreamerVariant& speech_streamer);

    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<ov::Tensor>& audios,
                                const OmniSpeechGenerationConfig& speech_config,
                               const StreamerVariant& text_streamer,
                               const OmniSpeechStreamerVariant& speech_streamer);

    ov::Tensor get_speaker_embedding(const std::string& name) const {
        return m_talker->get_speaker_embedding(name);
    }

    std::vector<std::string> list_speakers() const {
        return m_talker->list_speakers();
    }

private:
    /// @brief Reject non-Omni-capable models early, before any inference work.
    void assert_omni_capable() const;

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
    std::shared_ptr<TalkerBase> m_talker;
};

}  // namespace ov::genai
