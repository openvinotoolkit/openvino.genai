// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

#include "streaming_session_impl_base.hpp"

namespace ov::genai {

ASRStreamingSession::ASRStreamingSession(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) {}

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
