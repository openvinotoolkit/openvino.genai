// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>
#include <atomic>
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "synchronized_queue.hpp"

namespace ov::genai {
class GenerationStream {
    mutable std::mutex m_mutex;
    GenerationStatus m_status = GenerationStatus::RUNNING;
    GenerationFinishReason m_finish_reason = GenerationFinishReason::NONE;
    SynchronizedQueue<GenerationOutputs> m_output_queue;
    std::optional<PerfMetrics> m_perf_metrics;

public:
    using Ptr = std::shared_ptr<GenerationStream>;

    // Don't use directly
    GenerationStream() = default;

    static GenerationStream::Ptr create() {
        return std::make_shared<GenerationStream>();
    }

    void push(GenerationOutputs outputs) {
        m_output_queue.push(std::move(outputs));
    }

    GenerationOutputs read() {
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

    void set_perf_metrics(PerfMetrics perf_metrics) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_perf_metrics = std::move(perf_metrics);
    }

    PerfMetrics get_perf_metrics() {
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
}
