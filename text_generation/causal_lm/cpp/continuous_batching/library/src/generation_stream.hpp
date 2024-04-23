// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>
#include <atomic>
#include "continuous_batching_pipeline.hpp"
#include "synchronized_queue.hpp"
#include "generation_handle.hpp"


class GenerationStream {
    std::mutex m_mutex;
    bool m_handle_dropped = false;
    bool m_generation_finished = false;
    GenerationResultStatus m_finish_status;
    SynchronizedQueue<GenerationOutputs> m_output_queue;

    std::vector<uint64_t> last_sequence_ids;

public:
    using Ptr = std::shared_ptr<GenerationStream>;

    // Don't use directly
    GenerationStream() = default;

    static GenerationStream::Ptr create() {
        return std::make_shared<GenerationStream>();
    }

    void push(GenerationOutputs outputs) {
        m_output_queue.push(outputs);
    }

    // Retriving vector of pairs <sequence_id, token_id> as we can generate multiple outputs for a single prompt
    GenerationOutputs read() {
        return m_output_queue.pull();
    }

    bool can_read() {
        return !m_output_queue.empty();
    }

    void finish_generation_stream(GenerationResultStatus status) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_generation_finished = true;
        m_finish_status = status;
    }

    bool generation_finished() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_generation_finished;
    }

    GenerationResultStatus get_finish_status() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_finish_status;
    }

    void drop() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_handle_dropped = true;
    }

    bool handle_dropped() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_handle_dropped;
    }
};
