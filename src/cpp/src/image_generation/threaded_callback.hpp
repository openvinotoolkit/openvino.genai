// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thread>

#include "synchronized_queue.hpp"

namespace ov {
namespace genai {

enum class CallbackStatus {
    RUNNING = 0, // Continue to run callback
    STOP = 1 // Stop callback
};

class ThreadedCallbackWrapper {
public:
    ThreadedCallbackWrapper(std::function<bool(size_t, size_t, ov::Tensor&)> callback)
        : m_callback{callback} {}

    void start() {
        if (!m_callback) {
            return;
        }

        m_worker_thread = std::make_shared<std::thread>(&ThreadedCallbackWrapper::_worker, this);
    }

    CallbackStatus write(const size_t step, const size_t num_steps, const ov::Tensor& latent) {
        if (!m_callback || m_status == CallbackStatus::STOP) {
            return CallbackStatus::STOP;
        }

        m_squeue.push({step, num_steps, latent});

        return CallbackStatus::RUNNING;
    }

    void end() {
        if (!m_callback) {
            return;
        }

        m_status = CallbackStatus::STOP;
        m_squeue.empty();

        if (m_worker_thread && m_worker_thread->joinable()) {
            m_worker_thread->join();
        }
    }

    bool has_callback() const {
        return static_cast<bool>(m_callback);
    }

private:
    std::function<bool(size_t, size_t, ov::Tensor&)> m_callback = nullptr;
    std::shared_ptr<std::thread> m_worker_thread = nullptr;
    SynchronizedQueue<std::tuple<size_t, size_t, ov::Tensor>> m_squeue;

    std::atomic<CallbackStatus> m_status = CallbackStatus::RUNNING;

    void _worker() {
        while (m_status == CallbackStatus::RUNNING) {
            // wait for queue pull
            auto [step, num_steps, latent] = m_squeue.pull();

            if (m_callback(step, num_steps, latent)) {
                m_status = CallbackStatus::STOP;
                m_squeue.empty();
            }
        }
    }
};

}  // namespace genai
}  // namespace ov