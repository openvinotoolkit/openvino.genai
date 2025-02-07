// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thread>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "synchronized_queue.hpp"
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
        if (!m_streamer_ptr || m_dropped) {
            return;
        }

        for (const auto token : tokens) {
            m_squeue.push(token);
        }
    }

    void put(const int64_t token) {
        if (!m_streamer_ptr || m_dropped) {
            return;
        }

        m_squeue.push(token);
    }

    void end() {
        if (!m_streamer_ptr) {
            return;
        }

        // push stop token to unblock squeue.pull
        m_squeue.push(std::monostate());

        if (m_worker_thread && m_worker_thread->joinable()) {
            m_worker_thread->join();
        }

        m_streamer_ptr->end();
    }

    bool is_dropped() const {
        if (!m_streamer_ptr) {
            return false;
        }

        return m_dropped;
    }

    bool has_callback() const {
        return static_cast<bool>(m_streamer_ptr);
    }

private:
    std::shared_ptr<StreamerBase> m_streamer_ptr = nullptr;
    std::shared_ptr<std::thread> m_worker_thread = nullptr;
    SynchronizedQueue<std::variant<int64_t, std::monostate>> m_squeue;

    std::atomic<bool> m_dropped = false;

    void _worker() {
        while (true) {
            // wait for queue pull
            std::variant<int64_t, std::monostate> token_variant = m_squeue.pull();

            // wait for streamer_ptr result
            if (auto token = std::get_if<int64_t>(&token_variant)) {
                m_dropped = m_streamer_ptr->put(*token);
            } else if (auto stop_token = std::get_if<std::monostate>(&token_variant)) {
                break;
            } else {
                OPENVINO_THROW("Internal error: unsupported threaded streamer value");
            }

            if (m_dropped) {
                break;
            }
        }
    }
};

}  // namespace genai
}  // namespace ov
