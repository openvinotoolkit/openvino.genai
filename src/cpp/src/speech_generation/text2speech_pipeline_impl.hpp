// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>

#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "openvino/genai/speech_generation/text2speech_streaming_session.hpp"
#include "speech_generation/text2speech_streaming_session_impl.hpp"

namespace ov {
namespace genai {

class Text2SpeechPipelineImpl {
public:
    GenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const GenerationConfig& generation_config) {
        m_generation_config = generation_config;
        m_generation_config.validate();
    }

    virtual Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                               const ov::Tensor& speaker_embedding,
                                               const SpeechGenerationConfig& generation_config) = 0;

    /// @brief Streaming variant: audio chunks are delivered to `streamer` while generation runs.
    /// Backends without audio-streaming support reject an active streamer.
    virtual Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                               const ov::Tensor& speaker_embedding,
                                               const SpeechGenerationConfig& generation_config,
                                               const OmniSpeechStreamerVariant& streamer) {
        OPENVINO_ASSERT(std::holds_alternative<std::monostate>(streamer),
                        "Audio output streaming is not supported by this text-to-speech backend.");
        return generate(texts, speaker_embedding, generation_config);
    }

    /// @brief Number of streaming sessions (Text2SpeechStreamingSession) the backend can serve at the same time.
    /// Backends without input-streaming support allow none.
    virtual size_t max_streaming_sessions() const {
        return 0;
    }

    /// @brief Creates the backend side of a Text2SpeechStreamingSession. `owner` is this implementation, shared, so
    /// the session can keep it alive. Backends that support sessions take a slot with acquire_streaming_session_slot().
    virtual std::unique_ptr<Text2SpeechStreamingSessionImpl> create_streaming_session_impl(
        std::shared_ptr<Text2SpeechPipelineImpl> owner,
        const ov::Tensor& speaker_embedding,
        const SpeechGenerationConfig& generation_config,
        const Text2SpeechStreamingConfig& streaming_config) {
        OPENVINO_THROW("Input streaming (Text2SpeechStreamingSession) is not supported by this text-to-speech backend. "
                       "For Qwen3-TTS use the SDPA backend (ATTENTION_BACKEND = SDPA).");
    }

    Text2SpeechStreamingSessionSlot acquire_streaming_session_slot() {
        const size_t limit = max_streaming_sessions();
        size_t live = m_live_streaming_sessions.load();
        do {
            OPENVINO_ASSERT(live < limit,
                            "This text-to-speech backend supports ", limit,
                            " concurrent streaming session(s). Destroy the existing session first.");
        } while (!m_live_streaming_sessions.compare_exchange_weak(live, live + 1));
        return Text2SpeechStreamingSessionSlot{&m_live_streaming_sessions};
    }

    virtual SpeechGenerationPerfMetrics get_performance_metrics();

    /// @brief Get the expected speaker embedding shape for this backend.
    virtual ov::Shape get_speaker_embedding_shape() const = 0;

    virtual ~Text2SpeechPipelineImpl() = default;

protected:
    void save_load_time(std::chrono::steady_clock::time_point start_time);

    GenerationConfig m_generation_config;
    std::atomic<size_t> m_live_streaming_sessions{0};
    float m_load_time_ms = 0.0f;
    SpeechGenerationPerfMetrics m_perf_metrics;
};

}  // namespace genai
}  // namespace ov
