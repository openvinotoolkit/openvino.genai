// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thread>
#include <variant>

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

        m_squeue.push(std::make_tuple(step, num_steps, latent));

        return CallbackStatus::RUNNING;
    }

    void end() {
        if (!m_callback) {
            return;
        }

        m_status = CallbackStatus::STOP;
        m_squeue.push(std::monostate());

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
    SynchronizedQueue<std::variant<std::tuple<size_t, size_t, ov::Tensor>, std::monostate>> m_squeue;

    std::atomic<CallbackStatus> m_status = CallbackStatus::RUNNING;

    void _worker() {
        while (m_status == CallbackStatus::RUNNING) {
            auto item = m_squeue.pull();

            if (auto callback_data = std::get_if<std::tuple<size_t, size_t, ov::Tensor>>(&item)) {
                auto& [step, num_steps, latent] = *callback_data;
                const auto should_stop = m_callback(step, num_steps, latent);
                
                if (should_stop) {
                    m_status = CallbackStatus::STOP;
                }
            } else if (std::get_if<std::monostate>(&item)) {
                break;
            }
        }
    }
};

}  // namespace genai
}  // namespace ov