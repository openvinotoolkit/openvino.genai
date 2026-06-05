// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/pipeline_base.hpp"

#include "utils.hpp"
#include "visual_language/vision_properties.hpp"

namespace ov {
namespace genai {

VLMPipeline::VLMPipelineBase::VLMPipelineBase() : m_attention_backend(SDPA_BACKEND) {}

VLMPipeline::VLMPipelineBase::~VLMPipelineBase() = default;

GenerationConfig VLMPipeline::VLMPipelineBase::resolve_generation_config(const ov::AnyMap& config_map) {
    ov::genai::OptionalGenerationConfig optional_config = utils::get_config_from_map(config_map);
    GenerationConfig config = optional_config.value_or(get_generation_config());
    config.update_generation_config(config_map);
    return config;
}

std::vector<ov::Tensor> VLMPipeline::VLMPipelineBase::extract_audios_from_config_map(
    const ov::AnyMap& config_map) {
    std::vector<ov::Tensor> result;
    const auto audios_it = config_map.find(ov::genai::audios.name());
    if (audios_it == config_map.end()) {
        return result;
    }
    if (audios_it->second.is<std::vector<ov::Tensor>>()) {
        result = audios_it->second.as<std::vector<ov::Tensor>>();
    } else if (audios_it->second.is<ov::Tensor>()) {
        result = {audios_it->second.as<ov::Tensor>()};
    } else if (!audios_it->second.empty()) {
        OPENVINO_THROW("Property 'audios' must be ov::Tensor or std::vector<ov::Tensor>, got unsupported type.");
    }
    return result;
}

VLMDecodedResults VLMPipeline::VLMPipelineBase::generate(const std::string& prompt, const ov::AnyMap& config_map) {
    const auto vision_properties = extract_vision_properties(config_map);
    m_pending_audios = extract_audios_from_config_map(config_map);
    m_pending_speech_streamer = utils::get_audio_streamer_from_map(config_map);
    GenerationConfig config = resolve_generation_config(config_map);
    AudioStreamerGuard guard{m_pending_speech_streamer, m_pending_audios};
    return generate(prompt,
                    vision_properties.images.value_or(std::vector<ov::Tensor>{}),
                    vision_properties.videos.value_or(std::vector<ov::Tensor>{}),
                    vision_properties.videos_metadata.value_or(std::vector<VideoMetadata>{}),
                    config,
                    utils::get_streamer_from_map(config_map));
}

VLMDecodedResults VLMPipeline::VLMPipelineBase::generate(const ChatHistory& history, const ov::AnyMap& config_map) {
    const auto vision_properties = extract_vision_properties(config_map);
    m_pending_audios = extract_audios_from_config_map(config_map);
    m_pending_speech_streamer = utils::get_audio_streamer_from_map(config_map);
    GenerationConfig config = resolve_generation_config(config_map);
    AudioStreamerGuard guard{m_pending_speech_streamer, m_pending_audios};
    return generate(history,
                    vision_properties.images.value_or(std::vector<ov::Tensor>{}),
                    vision_properties.videos.value_or(std::vector<ov::Tensor>{}),
                    vision_properties.videos_metadata.value_or(std::vector<VideoMetadata>{}),
                    config,
                    utils::get_streamer_from_map(config_map));
}

VLMPipeline::VLMPipelineBase::HiddenStatesCollectionScope::HiddenStatesCollectionScope(VLMPipelineBase& pipeline)
    : m_pipeline(&pipeline) {
    m_pipeline->enable_hidden_states_collection(true);
}

VLMPipeline::VLMPipelineBase::HiddenStatesCollectionScope::~HiddenStatesCollectionScope() {
    if (m_pipeline) {
        m_pipeline->enable_hidden_states_collection(false);
    }
}

VLMPipeline::VLMPipelineBase::HiddenStatesCollectionScope::HiddenStatesCollectionScope(
    HiddenStatesCollectionScope&& other) noexcept
    : m_pipeline(other.m_pipeline) {
    other.m_pipeline = nullptr;
}

}  // namespace genai
}  // namespace ov
