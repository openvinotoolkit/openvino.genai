// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <optional>

#include "openvino/core/except.hpp"
#include "openvino/genai/speech_generation/speech_generation_handle.hpp"
#include "synchronized_queue.hpp"

namespace ov {
namespace genai {

// Single-session counterpart of GenerationStream (src/cpp/src/generation_stream.hpp): a TTS
// request never forks into multiple sequences, so outputs are queued directly rather than keyed
// by sequence id.
class SpeechGenerationStream {
    mutable std::mutex m_mutex;
    GenerationStatus m_status = GenerationStatus::RUNNING;
    GenerationFinishReason m_finish_reason = GenerationFinishReason::NONE;
    SynchronizedQueue<SpeechGenerationOutput> m_output_queue;
    std::optional<SpeechGenerationPerfMetrics> m_perf_metrics;

public:
    using Ptr = std::shared_ptr<SpeechGenerationStream>;

    // Don't use directly, use create() instead.
    SpeechGenerationStream() = default;

    static SpeechGenerationStream::Ptr create() {
        return std::make_shared<SpeechGenerationStream>();
    }

    void push(SpeechGenerationOutput output) {
        m_output_queue.push(std::move(output));
    }

    SpeechGenerationOutput read() {
        return m_output_queue.pull();
    }

    bool can_read() {
        return !m_output_queue.empty();
    }

    void set_generation_status(GenerationStatus status) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status = status;
    }

    GenerationStatus get_status() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_status;
    }

    void set_perf_metrics(SpeechGenerationPerfMetrics perf_metrics) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_perf_metrics = std::move(perf_metrics);
    }

    SpeechGenerationPerfMetrics get_perf_metrics() {
        std::lock_guard<std::mutex> lock(m_mutex);
        OPENVINO_ASSERT(m_perf_metrics.has_value(), "Perf metrics are not available until generation has completed.");
        return *m_perf_metrics;
    }

    GenerationFinishReason get_finish_reason() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_finish_reason;
    }

    void stop(GenerationFinishReason finish_reason = GenerationFinishReason::STOP) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status = GenerationStatus::STOP;
        m_finish_reason = finish_reason;
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status = GenerationStatus::CANCEL;
    }
};

}  // namespace genai
}  // namespace ov
