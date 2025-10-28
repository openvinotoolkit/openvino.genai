// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thread>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "synchronized_queue.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

enum class CallbackStatus {
    RUNNING = 0, // Continue to run callback
    STOP = 1 // Stop callback
};

class ThreadedCallbackWrapper {
public:
    ThreadedCallbackWrapper(std::function<bool(size_t, size_t, ov::Tensor&)> callback, const size_t num_steps)
        : m_callback{callback} {}

    void start() {
        if (!m_callback) {
            return;
        }

        m_worker_thread = std::make_shared<std::thread>(&ThreadedCallbackWrapper::_worker, this);
    }

    bool write(const size_t step, const size_t num_steps, const ov::Tensor& latent) {
        if (!m_callback || m_status == CallbackStatus::STOP) {
            return true;
        }

        m_squeue.push({step, num_steps, latent});

        return false;
    }

    void end() {
        if (!m_callback) {
            return;
        }

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
            std::tuple<size_t, size_t, ov::Tensor> intermediate_latent = m_squeue.pull();

            if (m_callback(std::get<0>(intermediate_latent), std::get<1>(intermediate_latent), std::get<2>(intermediate_latent))) {
                m_status == CallbackStatus::STOP;
            }
        }
    }
};

}  // namespace genai
}  // namespace ov