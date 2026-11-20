// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>

#include "openvino/genai/automatic_speech_recognition/streaming_session.hpp"

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "pipeline_base.hpp"
#include "streaming_session_impl_base.hpp"

namespace ov::genai {

namespace {

constexpr size_t SAMPLE_RATE = 16000;

void validate_streaming_config(const ASRStreamingConfig& config) {
    OPENVINO_ASSERT(config.chunk_size_sec > 0.0f,
                    "ASRStreamingConfig: chunk_size_sec must be > 0, got ",
                    config.chunk_size_sec);
    OPENVINO_ASSERT(config.window_chunk_num == 0 || config.window_chunk_num > config.unfixed_chunk_num,
                    "ASRStreamingConfig: window_chunk_num (",
                    config.window_chunk_num,
                    ") must be 0 (unbounded) or greater than unfixed_chunk_num (",
                    config.unfixed_chunk_num,
                    ")");
}

}  // namespace

ASRStreamingSessionImpl::ASRStreamingSessionImpl(size_t chunk_size_samples) : m_chunk_size_samples{chunk_size_samples} {
    OPENVINO_ASSERT(m_chunk_size_samples > 0, "ASR streaming chunk size must be positive.");
}

void ASRStreamingSessionImpl::attach(std::shared_ptr<ASRPipelineImplBase> owner,
                                       ASRStreamingSessionSlot slot,
                                       size_t chunk_size_samples) {
    if (m_chunk_size_samples == 0) {
        m_chunk_size_samples = chunk_size_samples;
    }
    OPENVINO_ASSERT(m_chunk_size_samples > 0, "ASR streaming chunk size must be positive.");
    m_owner = std::move(owner);
    m_slot = std::move(slot);
}

std::optional<ASRPartialResult> ASRStreamingSessionImpl::push_chunk(const std::vector<float>&) {
    OPENVINO_THROW("ASR streaming session implements neither process() nor push_chunk().");
}

ASRPartialResult ASRStreamingSessionImpl::finish() {
    OPENVINO_THROW("ASR streaming session implements neither process() nor finish().");
}

ASRStreamingResult ASRStreamingSessionImpl::process(const std::vector<float>& audio, bool is_last) {
    const auto start_time = std::chrono::steady_clock::now();
    std::optional<ASRPartialResult> partial;
    if (!audio.empty()) {
        partial = push_chunk(audio);
    }
    std::string new_committed_text = partial ? partial->new_committed_text : std::string{};
    if (is_last) {
        ASRPartialResult final_result = finish();
        new_committed_text += final_result.new_committed_text;
        partial = std::move(final_result);
    }

    ASRStreamingResult result;
    if (partial) {
        result.language = std::move(partial->language);
        result.committed_text = std::move(partial->committed_text);
        result.partial_text = std::move(partial->partial_text);
    }
    result.new_committed_text = std::move(new_committed_text);
    PerfMetrics::emplace_duration(result.perf_metrics.raw_metrics.generate_durations, start_time);
    result.perf_metrics.evaluate_statistics(start_time);
    return result;
}

void ASRStreamingSessionImpl::push(const std::vector<float>& pcm16k) {
    const std::lock_guard<std::mutex> lock{m_input_mutex};
    OPENVINO_ASSERT(!m_ended, "ASRStreamingSession::push() called after end().");
    m_buffer.insert(m_buffer.end(), pcm16k.begin(), pcm16k.end());
}

void ASRStreamingSessionImpl::end() {
    const std::lock_guard<std::mutex> lock{m_input_mutex};
    OPENVINO_ASSERT(!m_ended, "ASRStreamingSession::end() may only be called once.");
    m_ended = true;
}

std::optional<ASRStreamingResult> ASRStreamingSessionImpl::read() {
    const std::lock_guard<std::mutex> read_lock{m_read_mutex};
    if (m_final_returned) {
        return std::nullopt;
    }

    std::vector<float> audio;
    bool is_last = false;
    {
        const std::lock_guard<std::mutex> input_lock{m_input_mutex};
        if (m_ended && m_buffer.size() <= m_chunk_size_samples) {
            audio = std::move(m_buffer);
            m_buffer.clear();
            is_last = true;
        } else if (m_buffer.size() >= m_chunk_size_samples) {
            audio.assign(m_buffer.begin(), m_buffer.begin() + m_chunk_size_samples);
            m_buffer.erase(m_buffer.begin(), m_buffer.begin() + m_chunk_size_samples);
        } else {
            return std::nullopt;
        }
    }

    ASRStreamingResult result = process(audio, is_last);
    if (is_last) {
        result.is_final = true;
        m_final_returned = true;
    }
    return result;
}

ASRStreamingIncrementalSessionImpl::ASRStreamingIncrementalSessionImpl(std::unique_ptr<ASRStreamImplBase> stream,
                                                                       size_t chunk_size_samples)
    : ASRStreamingSessionImpl{chunk_size_samples},
      m_stream{std::move(stream)} {
    OPENVINO_ASSERT(m_stream, "ASR stream implementation must not be null.");
}

ASRStreamingResult ASRStreamingIncrementalSessionImpl::process(const std::vector<float>& audio, bool is_last) {
    const auto start_time = std::chrono::steady_clock::now();
    m_stream->m_perf_metrics = ASRPerfMetrics{};
    m_stream->m_perf_metrics.raw_metrics.m_inference_durations.emplace_back(MicroSeconds{0.0f});

    std::string delta = audio.empty() ? std::string{} : m_stream->process_audio(audio);
    if (is_last) {
        delta += m_stream->flush();
    }
    m_committed_text += delta;

    ASRStreamingResult result;
    result.committed_text = m_committed_text;
    result.new_committed_text = std::move(delta);
    result.perf_metrics = std::move(m_stream->m_perf_metrics);
    PerfMetrics::emplace_duration(result.perf_metrics.raw_metrics.generate_durations, start_time);
    result.perf_metrics.evaluate_statistics(start_time);
    return result;
}

ASRStreamingSession::ASRStreamingSession(ASRPipeline& pipeline,
                                         const ASRStreamingConfig& streaming_config,
                                         const std::optional<ASRGenerationConfig>& generation_config) {
    validate_streaming_config(streaming_config);
    std::shared_ptr<ASRPipelineImplBase> owner = pipeline.m_impl;
    ASRStreamingSessionSlot slot = owner->acquire_streaming_session_slot();
    const ASRGenerationConfig resolved = generation_config.value_or(pipeline.get_generation_config());
    m_impl = owner->create_streaming_session_impl(streaming_config, resolved);
    m_impl->attach(std::move(owner),
                   std::move(slot),
                   static_cast<size_t>(streaming_config.chunk_size_sec * SAMPLE_RATE));
}

ASRStreamingSession::~ASRStreamingSession() = default;

ASRStreamingSession::ASRStreamingSession(ASRStreamingSession&&) noexcept = default;

ASRStreamingSession& ASRStreamingSession::operator=(ASRStreamingSession&&) noexcept = default;

void ASRStreamingSession::push(const std::vector<float>& pcm16k) {
    OPENVINO_ASSERT(m_impl, "ASRStreamingSession has been moved from.");
    m_impl->push(pcm16k);
}

void ASRStreamingSession::end() {
    OPENVINO_ASSERT(m_impl, "ASRStreamingSession has been moved from.");
    m_impl->end();
}

std::optional<ASRStreamingResult> ASRStreamingSession::read() {
    OPENVINO_ASSERT(m_impl, "ASRStreamingSession has been moved from.");
    return m_impl->read();
}

}  // namespace ov::genai
