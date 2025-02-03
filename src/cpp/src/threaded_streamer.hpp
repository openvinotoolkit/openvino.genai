// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <queue>
#include <thread>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "text_callback_streamer.hpp"

namespace ov {
namespace genai {

class ThreadedStreamerWrapper {
public:
    ThreadedStreamerWrapper(const StreamerVariant& streamer, Tokenizer& tokenizer) {
        if (auto streamer_obj = std::get_if<std::monostate>(&streamer)) {
            m_streamer_ptr = nullptr;
        } else if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
            m_streamer_ptr = *streamer_obj;
        } else if (auto callback = std::get_if<std::function<bool(std::string)>>(&streamer)) {
            m_streamer_ptr = std::make_shared<TextCallbackStreamer>(tokenizer, *callback);
        }
    }

    void start() {
        if (!m_streamer_ptr) {
            return;
        }

        m_worker_thread = std::make_shared<std::thread>(&ThreadedStreamerWrapper::_worker, this);
    }

    void put(const std::vector<int64_t>& tokens) {
        if (!m_streamer_ptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(tokens);
        m_cv.notify_one();
    }

    void put(const int64_t token) {
        if (!m_streamer_ptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(token);
        m_cv.notify_one();
    }

    void end() {
        if (!m_streamer_ptr) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopped = true;
        }

        m_cv.notify_one();

        if (m_worker_thread && m_worker_thread->joinable()) {
            m_worker_thread->join();
        }

        m_streamer_ptr->end();
    }

    bool is_dropped() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_dropped;
    }

    bool has_callback() {
        return static_cast<bool>(m_streamer_ptr);
    }

private:
    std::shared_ptr<StreamerBase> m_streamer_ptr = nullptr;
    std::shared_ptr<std::thread> m_worker_thread = nullptr;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<std::variant<int64_t, std::vector<int64_t>>> m_queue;

    bool m_stopped = false;
    bool m_dropped = false;

    void _worker() {
        while (true) {
            std::variant<int64_t, std::vector<int64_t>> token_variant;
            {
                std::unique_lock<std::mutex> lock(m_mutex);

                // wait for the next token in queue or streamer stopped
                m_cv.wait(lock, [this] {
                    return m_stopped || !m_queue.empty();
                });

                // continue streaming until queue is empty
                if (m_stopped && m_queue.empty()) {
                    break;
                }

                token_variant = m_queue.front();
                m_queue.pop();
            }

            // wait for streamer_ptr result
            bool is_dropped = false;

            if (auto token = std::get_if<int64_t>(&token_variant)) {
                is_dropped = m_streamer_ptr->put(*token);
            } else {
                auto tokens = std::get_if<std::vector<int64_t>>(&token_variant);
                is_dropped = m_streamer_ptr->put(*tokens);
            }

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_dropped = is_dropped;

                if (m_dropped) {
                    break;
                }
            }
        }
    }
};

}  // namespace genai
}  // namespace ov
