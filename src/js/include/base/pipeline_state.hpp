// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "openvino/core/except.hpp"

enum class PipelineOperation { IDLE, INITIALIZE, GENERATE, DECODE, ACCESS };

struct PipelineState {
    std::mutex mutex;
    std::condition_variable decode_finished;
    PipelineOperation operation = PipelineOperation::IDLE;
    bool step_callback_active = false;
    bool decoding = false;
};

class PipelineOperationGuard {
public:
    PipelineOperationGuard(std::shared_ptr<PipelineState> state,
                           PipelineOperation operation,
                           const std::string& busy_message)
        : m_state(std::move(state)),
          m_operation(operation) {
        std::lock_guard<std::mutex> lock(m_state->mutex);
        if (operation == PipelineOperation::DECODE) {
            OPENVINO_ASSERT(!m_state->decoding &&
                                (m_state->operation == PipelineOperation::IDLE ||
                                 (m_state->operation == PipelineOperation::GENERATE &&
                                  m_state->step_callback_active)),
                            busy_message);
            m_state->decoding = true;
        } else {
            OPENVINO_ASSERT(m_state->operation == PipelineOperation::IDLE && !m_state->decoding, busy_message);
            m_state->operation = operation;
        }
    }

    PipelineOperationGuard(const PipelineOperationGuard&) = delete;
    PipelineOperationGuard& operator=(const PipelineOperationGuard&) = delete;

    ~PipelineOperationGuard() {
        std::lock_guard<std::mutex> lock(m_state->mutex);
        if (m_operation == PipelineOperation::DECODE) {
            m_state->decoding = false;
            m_state->decode_finished.notify_all();
        } else {
            m_state->operation = PipelineOperation::IDLE;
        }
    }

private:
    std::shared_ptr<PipelineState> m_state;
    PipelineOperation m_operation;
};

class PipelineStepGuard {
public:
    explicit PipelineStepGuard(std::shared_ptr<PipelineState> state) : m_state(std::move(state)) {
        std::lock_guard<std::mutex> lock(m_state->mutex);
        m_state->step_callback_active = true;
    }

    PipelineStepGuard(const PipelineStepGuard&) = delete;
    PipelineStepGuard& operator=(const PipelineStepGuard&) = delete;

    ~PipelineStepGuard() {
        std::unique_lock<std::mutex> lock(m_state->mutex);
        m_state->step_callback_active = false;
        m_state->decode_finished.wait(lock, [this] {
            return !m_state->decoding;
        });
    }

private:
    std::shared_ptr<PipelineState> m_state;
};
