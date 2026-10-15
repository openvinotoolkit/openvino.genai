// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "gguf_utils/gguf_lfm2_audio.hpp"
#include "omni/talker_speech_config_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/// @brief Implementation backing OmniPipeline. Holds a VLMPipelineBase (text-side) and a
/// TalkerBase (speech-side). Speech is gated per-call via OmniTalkerSpeechConfig::return_audio.
class OmniPipeline::OmniPipelineImpl {
public:
#ifdef ENABLE_GGUF
    OmniPipelineImpl(const std::filesystem::path& models, const std::string& device, const ov::AnyMap& properties)
        : m_lfm2(std::make_shared<GGUFLFM2Speech>(models, device, properties)) {}
#endif
    OmniPipelineImpl(const std::shared_ptr<VLMPipelineBase>& vlm,
                     const std::shared_ptr<TalkerBase>& talker);

    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<VideoMetadata>& videos_metadata,
                                const std::vector<ov::Tensor>& audios,
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer,
                                const OmniSpeechStreamerVariant& speech_streamer);

    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<VideoMetadata>& videos_metadata,
                                const std::vector<ov::Tensor>& audios,
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer,
                                const OmniSpeechStreamerVariant& speech_streamer);

    /// @brief Return the VLM's loaded GenerationConfig (parsed from generation_config.json).
    /// Used as the default text_config when callers don't pass one explicitly.
    GenerationConfig get_text_generation_config() const {
#ifdef ENABLE_GGUF
        if (m_lfm2)
            return m_lfm2->get_generation_config();
#endif
        return m_vlm->get_generation_config();
    }

    OmniTalkerSpeechConfig get_talker_speech_config() const {
        return m_default_talker_speech_config;
    }

    std::shared_ptr<VLMPipelineBase> get_vlm() const {
        OPENVINO_ASSERT(m_vlm, "This Omni backend has no standalone VLM; use OmniPipeline::generate");
        return m_vlm;
    }

    std::shared_ptr<TalkerBase> get_talker() const {
        OPENVINO_ASSERT(m_talker, "This Omni backend has no standalone talker; use OmniPipeline::generate");
        return m_talker;
    }

private:
    /// @brief Reject non-Omni-capable models early, before any inference work.
    void assert_omni_capable() const;

#ifdef ENABLE_GGUF
    std::shared_ptr<GGUFLFM2Speech> m_lfm2;
#endif
    std::shared_ptr<VLMPipelineBase> m_vlm;
    std::shared_ptr<TalkerBase> m_talker;
    OmniTalkerSpeechConfig m_default_talker_speech_config;
};

}  // namespace ov::genai
