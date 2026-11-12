// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

#include "pipeline_base.hpp"
#include "streaming_session_impl_base.hpp"

namespace ov::genai {

ASRStreamingSession::ASRStreamingSession(ASRPipeline& pipeline,
                                          const ASRStreamingConfig& streaming_config,
                                          const std::optional<ASRGenerationConfig>& generation_config) {
    OPENVINO_ASSERT(
        streaming_config.window_chunk_num == 0 ||
            streaming_config.window_chunk_num > streaming_config.window_rollback_chunk_num,
        "ASRStreamingConfig: window_chunk_num (",
        streaming_config.window_chunk_num,
        ") must be either 0 (disabled) or greater than window_rollback_chunk_num (",
        streaming_config.window_rollback_chunk_num,
        ")");
    const ASRGenerationConfig resolved = generation_config.value_or(pipeline.get_generation_config());
    m_impl = pipeline.m_impl->create_streaming_session_impl(streaming_config, resolved);
}

ASRStreamingSession::~ASRStreamingSession() = default;

ASRStreamingSession::ASRStreamingSession(ASRStreamingSession&&) noexcept = default;

ASRStreamingSession& ASRStreamingSession::operator=(ASRStreamingSession&&) noexcept = default;

std::optional<ASRPartialResult> ASRStreamingSession::push_chunk(const std::vector<float>& pcm16k) {
    OPENVINO_ASSERT(m_impl, "ASRStreamingSession has already been finished");
    return m_impl->push_chunk(pcm16k);
}

ASRPartialResult ASRStreamingSession::finish() {
    OPENVINO_ASSERT(m_impl, "ASRStreamingSession::finish() called more than once");
    ASRPartialResult result = m_impl->finish();
    m_impl.reset();
    return result;
}

}  // namespace ov::genai
