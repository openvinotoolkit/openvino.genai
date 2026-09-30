// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <thread>
#include <utility>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "synchronized_queue.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

class ThreadedStreamerWrapper {
public:
    ThreadedStreamerWrapper(const StreamerVariant& streamer, Tokenizer& tokenizer)
        : m_streamer_ptr{utils::create_streamer(streamer, tokenizer)} {}

    // Never rethrows: callers call end() on every path, so an error reaching here is already propagating.
    ~ThreadedStreamerWrapper() {
        finish();
    }

    void start() {
        if (!m_streamer_ptr) {
            return;
        }

        m_worker_thread = std::make_shared<std::thread>(&ThreadedStreamerWrapper::_worker, this);
    }

    void write(const std::vector<int64_t>& tokens) {
        if (!m_streamer_ptr || tokens.empty() || m_status != StreamingStatus::RUNNING) {
            return;
        }

        m_squeue.push(tokens);
    }

    void write(const int64_t token) {
        if (!m_streamer_ptr || m_status != StreamingStatus::RUNNING) {
            return;
        }

        m_squeue.push(token);
    }

    // Rethrows an exception raised by the streamer on the worker thread.
    void end() {
        finish();
        if (m_worker_exception) {
            std::rethrow_exception(std::exchange(m_worker_exception, nullptr));
        }
    }

    StreamingStatus get_status() const {
        return m_status;
    }

    bool has_callback() const {
        return static_cast<bool>(m_streamer_ptr);
    }

private:
    std::shared_ptr<StreamerBase> m_streamer_ptr = nullptr;
    std::shared_ptr<std::thread> m_worker_thread = nullptr;
    SynchronizedQueue<std::variant<int64_t, std::vector<int64_t>, std::monostate>> m_squeue;

    std::atomic<StreamingStatus> m_status = StreamingStatus::RUNNING;
    // Written by the worker before it exits and read only after join(), so no lock is needed.
    std::exception_ptr m_worker_exception = nullptr;
    bool m_finished = false;

    void finish() {
        if (!m_streamer_ptr || m_finished) {
            return;
        }
        m_finished = true;

        // push stop token to unblock squeue.pull
        m_squeue.push(std::monostate());

        if (m_worker_thread && m_worker_thread->joinable()) {
            m_worker_thread->join();
        }

        // A streamer that already failed is not asked to flush.
        if (!m_worker_exception) {
            m_streamer_ptr->end();
        }
    }

    void _worker() {
        // An exception escaping a std::thread calls std::terminate, so it is kept for end() instead.
        try {
            while (m_status == StreamingStatus::RUNNING) {
                // wait for queue pull
                std::variant<int64_t, std::vector<int64_t>, std::monostate> token_variant = m_squeue.pull();

                // wait for streamer_ptr result
                if (auto token = std::get_if<int64_t>(&token_variant)) {
                    m_status = _get_streaming_status(m_streamer_ptr->write(*token));
                } else if (auto tokens = std::get_if<std::vector<int64_t>>(&token_variant)) {
                    m_status = _get_streaming_status(m_streamer_ptr->write(*tokens));
                } else if (auto stop_token = std::get_if<std::monostate>(&token_variant)) {
                    break;
                } else {
                    OPENVINO_THROW("Internal error: unsupported threaded streamer value");
                }
            }
        } catch (...) {
            m_worker_exception = std::current_exception();
            // Cancelling makes the generation loop drop the request instead of finishing it.
            m_status = StreamingStatus::CANCEL;
        }
    }

    StreamingStatus _get_streaming_status(CallbackTypeVariant callback_status) {
        if (auto status = std::get_if<StreamingStatus>(&callback_status)) {
            return *status;
        } else {
            return std::get<bool>(callback_status) ? StreamingStatus::STOP : StreamingStatus::RUNNING;
        }
    }
};

}  // namespace genai
}  // namespace ov
