// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>

#include "openvino/core/except.hpp"
#include "openvino/genai/speech_generation/speech_generation_handle.hpp"

namespace ov {
namespace genai {

// Single-session counterpart of GenerationStream (src/cpp/src/generation_stream.hpp): a TTS
// request never forks into multiple sequences, so outputs are queued directly rather than keyed
// by sequence id.
class SpeechGenerationStream {
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    GenerationStatus m_status = GenerationStatus::RUNNING;
    GenerationFinishReason m_finish_reason = GenerationFinishReason::NONE;
    std::queue<SpeechGenerationOutput> m_output_queue;
    std::exception_ptr m_error;
    bool m_closed = false;
    std::function<void(GenerationStatus, GenerationFinishReason)> m_terminal_callback;
    std::optional<SpeechGenerationPerfMetrics> m_perf_metrics;

public:
    using Ptr = std::shared_ptr<SpeechGenerationStream>;

    // Don't use directly, use create() instead.
    SpeechGenerationStream() = default;

    static SpeechGenerationStream::Ptr create() {
        return std::make_shared<SpeechGenerationStream>();
    }

    void push(SpeechGenerationOutput output) {
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            if (m_closed) {
                return;
            }
            m_output_queue.push(std::move(output));
        }
        m_cv.notify_one();
    }

    void push_and_close(SpeechGenerationOutput output, GenerationStatus status) {
        OPENVINO_ASSERT(status == GenerationStatus::FINISHED || status == GenerationStatus::IGNORED,
                        "Speech output can only close successfully or with an out-of-memory status.");
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            if (m_closed) {
                return;
            }
            m_output_queue.push(std::move(output));
            m_status = status;
            m_closed = true;
        }
        m_cv.notify_all();
    }

    void close(GenerationStatus status) {
        OPENVINO_ASSERT(status == GenerationStatus::IGNORED,
                        "Closing a speech stream without output is only supported for out-of-memory.");
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            if (m_closed) {
                return;
            }
            m_status = status;
            m_closed = true;
        }
        m_cv.notify_all();
    }

    void fail(std::exception_ptr error) {
        OPENVINO_ASSERT(error, "Speech generation failure must contain an exception.");
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            if (m_closed) {
                return;
            }
            m_error = std::move(error);
            m_status = GenerationStatus::FAILED;
            m_closed = true;
        }
        m_cv.notify_all();
    }

    void set_terminal_callback(std::function<void(GenerationStatus, GenerationFinishReason)> callback) {
        std::lock_guard<std::mutex> lock{m_mutex};
        m_terminal_callback = std::move(callback);
    }

    SpeechGenerationOutput read() {
        std::unique_lock<std::mutex> lock{m_mutex};
        m_cv.wait(lock, [this] {
            return !m_output_queue.empty() || m_error || m_closed;
        });
        if (m_status == GenerationStatus::STOP || m_status == GenerationStatus::CANCEL) {
            SpeechGenerationOutput output;
            output.finish_reason = m_finish_reason;
            return output;
        }
        if (!m_output_queue.empty()) {
            SpeechGenerationOutput output = std::move(m_output_queue.front());
            m_output_queue.pop();
            return output;
        }
        if (m_error) {
            std::rethrow_exception(m_error);
        }
        return {};
    }

    bool can_read() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return !m_output_queue.empty() || static_cast<bool>(m_error);
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
        std::function<void(GenerationStatus, GenerationFinishReason)> callback;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_closed) {
                return;
            }
            m_status = GenerationStatus::STOP;
            m_finish_reason = finish_reason;
            m_closed = true;
            m_output_queue = {};
            callback = m_terminal_callback;
        }
        m_cv.notify_all();
        if (callback) {
            callback(GenerationStatus::STOP, finish_reason);
        }
    }

    void cancel() {
        std::function<void(GenerationStatus, GenerationFinishReason)> callback;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_closed) {
                return;
            }
            m_status = GenerationStatus::CANCEL;
            m_closed = true;
            m_output_queue = {};
            callback = m_terminal_callback;
        }
        m_cv.notify_all();
        if (callback) {
            callback(GenerationStatus::CANCEL, GenerationFinishReason::NONE);
        }
    }
};

}  // namespace genai
}  // namespace ov
