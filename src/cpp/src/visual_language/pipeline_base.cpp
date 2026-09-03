// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/pipeline_base.hpp"

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

VLMDecodedResults VLMPipeline::VLMBackend::generate(const std::string& prompt, const ov::AnyMap& config_map) {
    const auto multimodal_inputs = extract_multimodal_inputs(config_map);
    m_pending_speech_streamer = utils::get_audio_streamer_from_map(config_map);
    GenerationConfig config = resolve_generation_config(config_map);
    AudioStreamerGuard guard{m_pending_speech_streamer};
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
    m_pending_speech_streamer = utils::get_audio_streamer_from_map(config_map);
    GenerationConfig config = resolve_generation_config(config_map);
    AudioStreamerGuard guard{m_pending_speech_streamer};
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
