// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "omni/pipeline_impl.hpp"

#include <utility>

#include "openvino/core/except.hpp"

namespace ov::genai {

OmniPipeline::OmniPipelineImpl::OmniPipelineImpl(const std::shared_ptr<VLMPipeline::VLMPipelineBase>& vlm,
                                                  const std::shared_ptr<TalkerBase>& talker) {
    OPENVINO_ASSERT(vlm != nullptr, "OmniPipeline: VLM pointer is null");
    OPENVINO_ASSERT(talker != nullptr, "OmniPipeline: talker pointer is null");
    m_vlm = vlm;
    m_talker = talker;
    assert_omni_capable();
}

void OmniPipeline::OmniPipelineImpl::assert_omni_capable() const {
    OPENVINO_ASSERT(
        m_vlm->is_audio_output_enabled(),
        "OmniPipeline requires a Qwen3-Omni model with audio output enabled (config.json: enable_audio_output=true)");
    OPENVINO_ASSERT(
        m_vlm->supports_hidden_states_collection(),
        "OmniPipeline speech output requires the continuous-batching backend, but the loaded VLM uses the SDPA "
        "fallback path. Load the model with attention_backend=PA on a CPU or GPU device (NPU is not supported "
        "for Qwen3-Omni speech output).");
}

OmniDecodedResults OmniPipeline::OmniPipelineImpl::generate(const std::string& prompt,
                                                             const std::vector<ov::Tensor>& images,
                                                             const std::vector<ov::Tensor>& videos,
                                                             const std::vector<ov::Tensor>& audios,
                                                             const OmniSpeechGenerationConfig& speech_config,
                                                             const StreamerVariant& text_streamer,
                                                             const OmniSpeechStreamerVariant& speech_streamer) {
    speech_config.validate();

    PendingAudiosGuard audios_guard(*m_vlm, audios);
    GenerationConfig text_cfg = static_cast<GenerationConfig>(speech_config);

    if (speech_config.return_audio) {
        VLMPipeline::VLMPipelineBase::HiddenStatesCollectionScope hs_scope(*m_vlm);
        VLMDecodedResults vlm_result =
            m_vlm->generate(prompt, images, videos, /*videos_metadata=*/{}, text_cfg, text_streamer);
        return m_talker->generate(std::move(vlm_result), speech_config, speech_streamer);
    }

    // Text-only path: convert VLMDecodedResults to OmniDecodedResults with empty speech_outputs.
    VLMDecodedResults vlm_result = m_vlm->generate(prompt, images, videos, /*videos_metadata=*/{}, text_cfg, text_streamer);
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    return omni_result;
}

OmniDecodedResults OmniPipeline::OmniPipelineImpl::generate(const ChatHistory& history,
                                                             const std::vector<ov::Tensor>& images,
                                                             const std::vector<ov::Tensor>& videos,
                                                             const std::vector<ov::Tensor>& audios,
                                                             const OmniSpeechGenerationConfig& speech_config,
                                                             const StreamerVariant& text_streamer,
                                                             const OmniSpeechStreamerVariant& speech_streamer) {
    speech_config.validate();

    PendingAudiosGuard audios_guard(*m_vlm, audios);
    GenerationConfig text_cfg = static_cast<GenerationConfig>(speech_config);

    if (speech_config.return_audio) {
        // Speech output requires the prompts overload's per-prompt loop, which captures the
        // prompt-id slice into VLMDecodedResults::m_hidden_states_data. The histories overload
        // doesn't yet wire that capture (CB pipeline_base.cpp leaves original_prompt_ids_list
        // empty for the ChatHistory path — see the FIXME there). Apply the chat template here
        // and route through the prompts path so speech generation has its prompt_ids.
        const std::string templated_prompt = m_vlm->get_tokenizer().apply_chat_template(history, true);
        text_cfg.apply_chat_template = false;
        VLMPipeline::VLMPipelineBase::HiddenStatesCollectionScope hs_scope(*m_vlm);
        VLMDecodedResults vlm_result =
            m_vlm->generate(templated_prompt, images, videos, /*videos_metadata=*/{}, text_cfg, text_streamer);
        return m_talker->generate(std::move(vlm_result), speech_config, speech_streamer);
    }

    // Text-only path: convert VLMDecodedResults to OmniDecodedResults with empty speech_outputs.
    VLMDecodedResults vlm_result = m_vlm->generate(history, images, videos, /*videos_metadata=*/{}, text_cfg, text_streamer);
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    return omni_result;
}

}  // namespace ov::genai
