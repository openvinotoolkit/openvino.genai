// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/pipeline.hpp"

#include "omni/pipeline_impl.hpp"

namespace ov::genai {

OmniPipeline::OmniPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties)
    : m_pimpl(std::make_unique<OmniPipelineImpl>(models_path, device, properties)) {}

OmniPipeline::OmniPipeline(const std::shared_ptr<VLMPipeline::VLMPipelineBase>& vlm,
                           const std::filesystem::path& speech_models_path,
                           const std::string& device,
                           const ov::AnyMap& properties)
    : m_pimpl(std::make_unique<OmniPipelineImpl>(vlm, speech_models_path, device, properties)) {}

OmniPipeline::~OmniPipeline() = default;
OmniPipeline::OmniPipeline(OmniPipeline&&) noexcept = default;
OmniPipeline& OmniPipeline::operator=(OmniPipeline&&) noexcept = default;

OmniDecodedResults OmniPipeline::generate(const std::string& prompt,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const std::vector<ov::Tensor>& audios,
                                           const OmniSpeechGenerationConfig& speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt, images, videos, audios, speech_config, text_streamer, speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const std::vector<ov::Tensor>& audios,
                                           const OmniSpeechGenerationConfig& speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history, images, videos, audios, speech_config, text_streamer, speech_streamer);
}

}  // namespace ov::genai
