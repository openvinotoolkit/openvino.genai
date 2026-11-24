// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/text2speech_streaming_session.hpp"

#include "speech_generation/text2speech_pipeline_impl.hpp"
#include "speech_generation/text2speech_streaming_session_impl.hpp"

namespace ov::genai {

Text2SpeechStreamingSession::Text2SpeechStreamingSession(Text2SpeechPipeline& pipeline,
                                                         const ov::Tensor& speaker_embedding,
                                                         const std::optional<SpeechGenerationConfig>& generation_config,
                                                         const Text2SpeechStreamingConfig& streaming_config) {
    SpeechGenerationConfig config = generation_config.has_value() ? *generation_config : pipeline.m_speech_gen_config;
    config.validate();
    m_impl = pipeline.m_impl->create_streaming_session_impl(pipeline.m_impl, speaker_embedding, config, streaming_config);
}

Text2SpeechStreamingSession::~Text2SpeechStreamingSession() = default;

Text2SpeechStreamingSession::Text2SpeechStreamingSession(Text2SpeechStreamingSession&&) noexcept = default;

Text2SpeechStreamingSession& Text2SpeechStreamingSession::operator=(Text2SpeechStreamingSession&&) noexcept = default;

void Text2SpeechStreamingSession::push(const std::string& text) {
    OPENVINO_ASSERT(m_impl, "Text2SpeechStreamingSession has been moved from.");
    m_impl->push(text);
}

void Text2SpeechStreamingSession::end() {
    OPENVINO_ASSERT(m_impl, "Text2SpeechStreamingSession has been moved from.");
    m_impl->end();
}

std::optional<Text2SpeechStreamingResult> Text2SpeechStreamingSession::read() {
    OPENVINO_ASSERT(m_impl, "Text2SpeechStreamingSession has been moved from.");
    return m_impl->read();
}

}  // namespace ov::genai
