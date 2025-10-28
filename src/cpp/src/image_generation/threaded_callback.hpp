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

class ThreadedCallbackWrapper {
public:
    ThreadedCallbackWrapper(std::function<bool(size_t, size_t, ov::Tensor&)> callback, const size_t num_steps)
        : m_callback{callback}, m_num_steps{num_steps}, m_step{0} {}

    void start() {
        if (!m_callback) {
            return;
        }

        m_worker_thread = std::make_shared<std::thread>(&ThreadedCallbackWrapper::_worker, this);
    }

    bool write(const size_t step, const size_t num_steps, const ov::Tensor& latent) {
        if (!m_callback) {
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

    size_t m_num_steps;
    size_t m_step;

    void _worker() {
        while (m_step < (m_num_steps - 1)) {
            // wait for queue pull
            std::tuple<size_t, size_t, ov::Tensor> intermediate_latent = m_squeue.pull();

            m_callback(std::get<0>(intermediate_latent), std::get<1>(intermediate_latent), std::get<2>(intermediate_latent));
            // Update steps
            m_step = std::get<0>(intermediate_latent);
        }
    }
};

}  // namespace genai
}  // namespace ov