// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <queue>

#include "openvino/core/except.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
namespace ov::genai {
class GenerationStream {
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    GenerationStatus m_status = GenerationStatus::RUNNING;
    GenerationFinishReason m_finish_reason = GenerationFinishReason::NONE;
    std::queue<GenerationOutputs> m_output_queue;
    std::exception_ptr m_error;
    bool m_closed = false;
    std::optional<PerfMetrics> m_perf_metrics;
    std::optional<VLMPerfMetrics> m_vlm_perf_metrics;

    void clear_output_queue() noexcept {
        while (!m_output_queue.empty()) {
            m_output_queue.pop();
        }
    }

public:
    using Ptr = std::shared_ptr<GenerationStream>;

    // Don't use directly
    GenerationStream() = default;

    static GenerationStream::Ptr create() {
        return std::make_shared<GenerationStream>();
    }

    void push(GenerationOutputs outputs) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_closed) {
                return;
            }
            m_output_queue.push(std::move(outputs));
        }
        m_cv.notify_one();
    }

    void push_and_close(GenerationOutputs outputs, GenerationStatus status) {
        OPENVINO_ASSERT(status == GenerationStatus::FINISHED || status == GenerationStatus::IGNORED,
                        "Only successful or out-of-memory terminal statuses can publish terminal output.");
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_closed) {
                return;
            }
            m_output_queue.push(std::move(outputs));
            m_status = status;
            m_closed = true;
        }
        m_cv.notify_all();
    }

    GenerationOutputs read() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] {
            return !m_output_queue.empty() || m_error || m_closed;
        });
        if (m_status == GenerationStatus::STOP || m_status == GenerationStatus::CANCEL) {
            return {};
        }
        if (!m_output_queue.empty()) {
            GenerationOutputs outputs = std::move(m_output_queue.front());
            m_output_queue.pop();
            return outputs;
        }
        if (m_error) {
            std::rethrow_exception(m_error);
        }
        OPENVINO_THROW("Generation stream has reached end-of-stream.");
    }

    bool can_read() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return !m_output_queue.empty() || static_cast<bool>(m_error);
    }

    void set_generation_status(GenerationStatus status) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_status == GenerationStatus::FAILED || m_status == GenerationStatus::FINISHED ||
                m_status == GenerationStatus::IGNORED) {
                return;
            }
            m_status = status;
            m_closed = status != GenerationStatus::RUNNING;
            if (status == GenerationStatus::STOP || status == GenerationStatus::CANCEL) {
                m_output_queue = {};
            }
        }
        m_cv.notify_all();
    }

    void fail(std::exception_ptr error) {
        OPENVINO_ASSERT(error, "Generation failure must contain an exception.");
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_status == GenerationStatus::FINISHED || m_status == GenerationStatus::IGNORED ||
                m_status == GenerationStatus::FAILED) {
                return;
            }
            m_error = std::move(error);
            m_status = GenerationStatus::FAILED;
            m_closed = true;
        }
        m_cv.notify_all();
    }

    GenerationStatus get_status() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_status;
    }

    void set_perf_metrics(PerfMetrics perf_metrics) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_perf_metrics = std::move(perf_metrics);
    }

    PerfMetrics get_perf_metrics() {
        std::lock_guard<std::mutex> lock(m_mutex);
        OPENVINO_ASSERT(m_perf_metrics.has_value(), "Perf metrics are not available until generation has completed.");
        return *m_perf_metrics;
    }

    void set_vlm_perf_metrics(VLMPerfMetrics vlm_perf_metrics) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_vlm_perf_metrics = std::move(vlm_perf_metrics);
    }

    std::optional<VLMPerfMetrics> get_vlm_perf_metrics() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_vlm_perf_metrics;
    }

    GenerationFinishReason get_finish_reason() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_finish_reason;
    }

    void stop(GenerationFinishReason finish_reason = GenerationFinishReason::STOP) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_status != GenerationStatus::RUNNING) {
                return;
            }
            m_status = GenerationStatus::STOP;
            m_finish_reason = finish_reason;
            m_closed = true;
            clear_output_queue();
        }
        m_cv.notify_all();
    }

    void cancel() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_status != GenerationStatus::RUNNING) {
                return;
            }
            m_status = GenerationStatus::CANCEL;
            m_closed = true;
            clear_output_queue();
        }
        m_cv.notify_all();
    }
};
}
