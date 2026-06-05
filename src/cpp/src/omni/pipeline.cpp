// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni_pipeline.hpp"

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

VLMDecodedResults OmniPipeline::generate(const std::string& prompt,
                                          const std::vector<ov::Tensor>& images,
                                          const std::vector<ov::Tensor>& videos,
                                          const std::vector<ov::Tensor>& audios,
                                          const OmniSpeechGenerationConfig& speech_config,
                                          const StreamerVariant& text_streamer,
                                          const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt, images, videos, audios, speech_config, text_streamer, speech_streamer);
}

VLMDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                          const std::vector<ov::Tensor>& images,
                                          const std::vector<ov::Tensor>& videos,
                                          const std::vector<ov::Tensor>& audios,
                                          const OmniSpeechGenerationConfig& speech_config,
                                          const StreamerVariant& text_streamer,
                                          const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history, images, videos, audios, speech_config, text_streamer, speech_streamer);
}

void OmniPipeline::start_chat(const std::string& system_message) {
    m_pimpl->start_chat(system_message);
}

void OmniPipeline::finish_chat() {
    m_pimpl->finish_chat();
}

}  // namespace ov::genai
