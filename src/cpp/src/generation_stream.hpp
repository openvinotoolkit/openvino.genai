// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>
#include <atomic>
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "synchronized_queue.hpp"

namespace ov::genai {
class GenerationStream {
    std::mutex m_mutex;
    GenerationStatus m_status = GenerationStatus::RUNNING;
    SynchronizedQueue<GenerationOutputs> m_output_queue;

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

    // Retrieving vector of pairs <sequence_id, token_ids> as we can generate multiple outputs for a single prompt
    GenerationOutputs back() {
        return m_output_queue.back();
    }

    GenerationOutputs read() {
        return m_output_queue.pull();
    }

    bool can_read() {
        return !m_output_queue.empty() && !m_output_queue.full();
    }

    void set_generation_status(GenerationStatus status) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status = status;
    }

    GenerationStatus get_status() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_status;
    }

    void drop() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status = GenerationStatus::DROPPED_BY_HANDLE;
    }
};
}
