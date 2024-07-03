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
    GenerationStatus m_status = GenerationStatus::RUNNING;
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
        m_output_queue.push(std::move(outputs));
    }

    // Retriving vector of pairs <sequence_id, token_id> as we can generate multiple outputs for a single prompt
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

    void drop() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status = GenerationStatus::DROPPED_BY_HANDLE;
    }
};
