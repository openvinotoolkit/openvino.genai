// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/pipeline_base.hpp"

#include "omni/talker_speech_config_utils.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "utils.hpp"
#include "visual_language/multimodal_inputs.hpp"

namespace ov {
namespace genai {

VLMPipeline::VLMBackend::VLMBackend() : m_attention_backend(SDPA_BACKEND) {}

VLMPipeline::VLMBackend::~VLMBackend() = default;

GenerationConfig VLMPipeline::VLMBackend::resolve_generation_config(const ov::AnyMap& config_map) {
    ov::genai::OptionalGenerationConfig optional_config = utils::get_config_from_map(config_map);
    GenerationConfig config = optional_config.value_or(get_generation_config());
    config.update_generation_config(config_map);
    return config;
}

void VLMPipeline::VLMBackend::extract_speech_properties(const ov::AnyMap& config_map) {
    m_pending_speech_streamer = utils::get_speech_streamer_from_map(config_map);
    const auto it = config_map.find(talker_speech_config.name());
    if (it == config_map.end()) {
        return;
    }
    const auto speech_config = it->second.as<OmniTalkerSpeechConfig>();
    validate_omni_talker_speech_config(speech_config);
    if (speech_config.return_audio) {
        check_speech_in_stream(speech_config);
        m_pending_speech_config = speech_config;
    }
}

void VLMPipeline::VLMBackend::check_speech_in_stream(const OmniTalkerSpeechConfig&) const {
    OPENVINO_THROW("Speech output through talker_speech_config needs a VLM that decodes speech in its language model "
                   "stream: LFM3-Omni with ATTENTION_BACKEND=SDPA on CPU or GPU. Set return_audio=false for text only.");
}

VLMDecodedResults VLMPipeline::VLMBackend::generate(const std::string& prompt, const ov::AnyMap& config_map) {
    const auto multimodal_inputs = extract_multimodal_inputs(config_map);
    SpeechPropertiesGuard guard{m_pending_speech_streamer, m_pending_speech_config};
    extract_speech_properties(config_map);
    GenerationConfig config = resolve_generation_config(config_map);
    return generate(prompt,
                    multimodal_inputs.images.value_or(std::vector<ov::Tensor>{}),
                    multimodal_inputs.videos.value_or(std::vector<ov::Tensor>{}),
                    multimodal_inputs.audios.value_or(std::vector<ov::Tensor>{}),
                    multimodal_inputs.videos_metadata.value_or(std::vector<VideoMetadata>{}),
                    config,
                    utils::get_streamer_from_map(config_map));
}

VLMDecodedResults VLMPipeline::VLMBackend::generate(const ChatHistory& history, const ov::AnyMap& config_map) {
    const auto multimodal_inputs = extract_multimodal_inputs(config_map);
    SpeechPropertiesGuard guard{m_pending_speech_streamer, m_pending_speech_config};
    extract_speech_properties(config_map);
    GenerationConfig config = resolve_generation_config(config_map);
    return generate(history,
                    multimodal_inputs.images.value_or(std::vector<ov::Tensor>{}),
                    multimodal_inputs.videos.value_or(std::vector<ov::Tensor>{}),
                    multimodal_inputs.audios.value_or(std::vector<ov::Tensor>{}),
                    multimodal_inputs.videos_metadata.value_or(std::vector<VideoMetadata>{}),
                    config,
                    utils::get_streamer_from_map(config_map));
}

}  // namespace genai
}  // namespace ov
