// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/pipeline_base.hpp"

#include "omni/talker_speech_config_utils.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "utils.hpp"
#include "visual_language/multimodal_inputs.hpp"

namespace ov {
namespace genai {

namespace {
const OmniSpeechStreamerVariant NO_SPEECH_STREAMER = std::monostate{};
}  // namespace

VLMPipeline::VLMBackend::VLMBackend() : m_attention_backend(SDPA_BACKEND) {}

VLMPipeline::VLMBackend::~VLMBackend() = default;

GenerationConfig VLMPipeline::VLMBackend::resolve_generation_config(const ov::AnyMap& config_map) {
    ov::genai::OptionalGenerationConfig optional_config = utils::get_config_from_map(config_map);
    GenerationConfig config = optional_config.value_or(get_generation_config());
    config.update_generation_config(config_map);
    return config;
}

std::optional<VLMPipeline::VLMBackend::SpeechRequest> VLMPipeline::VLMBackend::read_speech_request(
    const ov::AnyMap& config_map) {
    const auto it = config_map.find(talker_speech_config.name());
    if (it == config_map.end()) {
        return std::nullopt;
    }
    auto speech_config = it->second.as<OmniTalkerSpeechConfig>();
    if (!speech_config.return_audio) {
        validate_omni_talker_speech_config(speech_config);
        return std::nullopt;
    }
    return SpeechRequest{std::move(speech_config), utils::get_speech_streamer_from_map(config_map)};
}

void VLMPipeline::VLMBackend::check_speech_request(const SpeechRequest& speech_request) const {
    validate_omni_talker_speech_config(speech_request.config);
    check_speech_in_stream(speech_request.config);
}

void VLMPipeline::VLMBackend::check_speech_in_stream(const OmniTalkerSpeechConfig&) const {
    OPENVINO_THROW("Speech output through talker_speech_config needs a VLM that decodes speech in its language model "
                   "stream: LFM3-Omni with ATTENTION_BACKEND=SDPA on CPU or GPU. Set return_audio=false for text only.");
}

VLMDecodedResults VLMPipeline::VLMBackend::generate_with_speech(const std::string& prompt,
                                                                const std::vector<ov::Tensor>& images,
                                                                const std::vector<ov::Tensor>& videos,
                                                                const std::vector<ov::Tensor>& audios,
                                                                const std::vector<VideoMetadata>& videos_metadata,
                                                                const GenerationConfig& generation_config,
                                                                const StreamerVariant& streamer,
                                                                const std::optional<SpeechRequest>&) {
    return generate(prompt, images, videos, audios, videos_metadata, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::VLMBackend::generate_with_speech(const ChatHistory& history,
                                                                const std::vector<ov::Tensor>& images,
                                                                const std::vector<ov::Tensor>& videos,
                                                                const std::vector<ov::Tensor>& audios,
                                                                const std::vector<VideoMetadata>& videos_metadata,
                                                                const GenerationConfig& generation_config,
                                                                const StreamerVariant& streamer,
                                                                const std::optional<SpeechRequest>&) {
    return generate(history, images, videos, audios, videos_metadata, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::VLMBackend::generate(const std::string& prompt, const ov::AnyMap& config_map) {
    const auto speech_request = read_speech_request(config_map);
    // Guarded before validation, so a rejected config still ends the speech streamer.
    SpeechStreamerGuard speech_streamer_guard(speech_request ? speech_request->streamer : NO_SPEECH_STREAMER,
                                              speech_request && is_speech_streamer_active(speech_request->streamer));
    if (speech_request) {
        check_speech_request(*speech_request);
    }
    const auto multimodal_inputs = extract_multimodal_inputs(config_map);
    GenerationConfig config = resolve_generation_config(config_map);
    VLMDecodedResults results =
        generate_with_speech(prompt,
                             multimodal_inputs.images.value_or(std::vector<ov::Tensor>{}),
                             multimodal_inputs.videos.value_or(std::vector<ov::Tensor>{}),
                             multimodal_inputs.audios.value_or(std::vector<ov::Tensor>{}),
                             multimodal_inputs.videos_metadata.value_or(std::vector<VideoMetadata>{}),
                             config,
                             utils::get_streamer_from_map(config_map),
                             speech_request);
    // The speech streamer is ended here, so an end() failure reaches the caller; the guard only logs it during
    // unwinding.
    speech_streamer_guard.end();
    return results;
}

VLMDecodedResults VLMPipeline::VLMBackend::generate(const ChatHistory& history, const ov::AnyMap& config_map) {
    const auto speech_request = read_speech_request(config_map);
    // Guarded before validation, so a rejected config still ends the speech streamer.
    SpeechStreamerGuard speech_streamer_guard(speech_request ? speech_request->streamer : NO_SPEECH_STREAMER,
                                              speech_request && is_speech_streamer_active(speech_request->streamer));
    if (speech_request) {
        check_speech_request(*speech_request);
    }
    const auto multimodal_inputs = extract_multimodal_inputs(config_map);
    GenerationConfig config = resolve_generation_config(config_map);
    VLMDecodedResults results =
        generate_with_speech(history,
                             multimodal_inputs.images.value_or(std::vector<ov::Tensor>{}),
                             multimodal_inputs.videos.value_or(std::vector<ov::Tensor>{}),
                             multimodal_inputs.audios.value_or(std::vector<ov::Tensor>{}),
                             multimodal_inputs.videos_metadata.value_or(std::vector<VideoMetadata>{}),
                             config,
                             utils::get_streamer_from_map(config_map),
                             speech_request);
    // The speech streamer is ended here, so an end() failure reaches the caller; the guard only logs it during
    // unwinding.
    speech_streamer_guard.end();
    return results;
}

}  // namespace genai
}  // namespace ov
