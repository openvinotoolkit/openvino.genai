// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "omni/pipeline_impl.hpp"

#include <utility>

#include "openvino/core/except.hpp"

namespace ov::genai {

namespace {

/// @brief Cross-config validation: when speech output is requested the thinker text decode
/// must use a sampling mode the talker can consume — single hidden-state stream, no beam
/// candidates, no speculative draft tokens.
void enforce_text_config_compatible_with_audio(const GenerationConfig& text_config,
                                               const OmniTalkerSpeechConfig& talker_speech_config) {
    if (!talker_speech_config.return_audio) {
        return;
    }
    OPENVINO_ASSERT(!text_config.is_beam_search(),
                    "OmniPipeline: return_audio is not compatible with beam search (num_beams > 1)");
    OPENVINO_ASSERT(!text_config.is_prompt_lookup() && !text_config.is_assisting_generation(),
                    "OmniPipeline: return_audio is not compatible with prompt lookup or assistant/speculative decoding");
    OPENVINO_ASSERT(text_config.num_return_sequences == 1,
                    "OmniPipeline: return_audio requires num_return_sequences == 1 (got ",
                    text_config.num_return_sequences,
                    "); the talker consumes a single hidden-state stream");
}

}  // namespace

OmniPipeline::OmniPipelineImpl::OmniPipelineImpl(const std::shared_ptr<VLMPipelineBase>& vlm,
                                                  const std::shared_ptr<TalkerBase>& talker) :
    m_vlm{vlm}, m_talker{talker} {
    OPENVINO_ASSERT(m_vlm, "OmniPipeline: VLM pointer is null");
    OPENVINO_ASSERT(m_talker, "OmniPipeline: talker pointer is null");
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
                                                             const std::vector<VideoMetadata>& videos_metadata,
                                                             const std::vector<ov::Tensor>& audios,
                                                             const GenerationConfig& text_config,
                                                             const OmniTalkerSpeechConfig& talker_speech_config,
                                                             const StreamerVariant& streamer,
                                                             const OmniSpeechStreamerVariant& speech_streamer) {
    validate_omni_talker_speech_config(talker_speech_config);
    enforce_text_config_compatible_with_audio(text_config, talker_speech_config);

    OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                    "OmniPipeline: videos_metadata size (", videos_metadata.size(),
                    ") must match videos size (", videos.size(), ") or be empty");

    if (talker_speech_config.return_audio) {
        GenerationConfig text_cfg = text_config;
        text_cfg.return_omni_outputs = true;
        VLMDecodedResults vlm_result =
            m_vlm->generate(prompt, images, videos, audios, videos_metadata, text_cfg, streamer);
        TalkerResults talker_result = m_talker->generate(vlm_result, talker_speech_config, speech_streamer);
        OmniDecodedResults omni_result;
        static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
        omni_result.speech_result = std::move(talker_result);
        return omni_result;
    }

    // Text-only path: convert VLMDecodedResults to OmniDecodedResults with empty speech_result.
    VLMDecodedResults vlm_result =
        m_vlm->generate(prompt, images, videos, audios, videos_metadata, text_config, streamer);
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    return omni_result;
}

OmniDecodedResults OmniPipeline::OmniPipelineImpl::generate(const ChatHistory& history,
                                                             const std::vector<ov::Tensor>& images,
                                                             const std::vector<ov::Tensor>& videos,
                                                             const std::vector<VideoMetadata>& videos_metadata,
                                                             const std::vector<ov::Tensor>& audios,
                                                             const GenerationConfig& text_config,
                                                             const OmniTalkerSpeechConfig& talker_speech_config,
                                                             const StreamerVariant& streamer,
                                                             const OmniSpeechStreamerVariant& speech_streamer) {
    validate_omni_talker_speech_config(talker_speech_config);
    enforce_text_config_compatible_with_audio(text_config, talker_speech_config);

    OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                    "OmniPipeline: videos_metadata size (", videos_metadata.size(),
                    ") must match videos size (", videos.size(), ") or be empty");

    if (talker_speech_config.return_audio) {
        // Speech output requires the prompts overload's per-prompt loop, which captures the
        // prompt-id slice into VLMDecodedResults::prompt_ids. The histories overload doesn't
        // yet wire that capture (CB pipeline_base.cpp leaves original_prompt_ids_list empty
        // for the ChatHistory path — see the FIXME there). Apply the chat template here and
        // route through the prompts path so speech generation has its prompt_ids.
        const std::string templated_prompt = m_vlm->get_tokenizer().apply_chat_template(history, true);
        GenerationConfig text_cfg = text_config;
        text_cfg.apply_chat_template = false;
        text_cfg.return_omni_outputs = true;
        VLMDecodedResults vlm_result =
            m_vlm->generate(templated_prompt, images, videos, audios, videos_metadata, text_cfg, streamer);
        TalkerResults talker_result = m_talker->generate(vlm_result, talker_speech_config, speech_streamer);
        OmniDecodedResults omni_result;
        static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
        omni_result.speech_result = std::move(talker_result);
        return omni_result;
    }

    // Text-only path: convert VLMDecodedResults to OmniDecodedResults with empty speech_result.
    VLMDecodedResults vlm_result =
        m_vlm->generate(history, images, videos, audios, videos_metadata, text_config, streamer);
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    return omni_result;
}

}  // namespace ov::genai
