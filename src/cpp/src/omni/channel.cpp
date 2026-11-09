// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/channel.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace ov {
namespace genai {

/// @brief Queue behind OmniChannel. Pimpl'd so <mutex> / <condition_variable> stay out of the
/// public header — a std::mutex member would otherwise cross the shared-library boundary.
class OmniChannel::Impl {
    /// @brief How the read end stands. Monotone toward the stronger state: detach() cannot soften a
    /// prior abort(), so a failure can never be downgraded by a talker still winding down.
    enum class ReaderState { OPEN, DETACHED, ABORTED };

public:
    StreamingStatus write(const ov::AnyMap& data) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_reader_state != ReaderState::OPEN) {
                // Dropping instead of queueing is the whole point: nobody is going to read this,
                // and the queue is unbounded.
                return m_reader_state == ReaderState::ABORTED ? StreamingStatus::STOP
                                                              : StreamingStatus::RUNNING;
            }
            m_queue.push_back(data);
        }
        m_cv.notify_one();
        return StreamingStatus::RUNNING;
    }

    void end() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_finished = true;
        }
        m_cv.notify_all();
    }

    void abandon() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_finished = true;
            m_truncated = true;
            // Unlike end(), which lets the reader drain first: the response is never going to be
            // completed, so draining only buys speech nobody will hear.
            m_queue.clear();
        }
        m_cv.notify_all();
    }

    std::optional<ov::AnyMap> read() {
        std::unique_lock<std::mutex> lock(m_mutex);
        // Queued steps win over m_finished, so end() never discards what was already written:
        // the reader keeps draining and only sees the end of the stream once the queue is empty.
        m_cv.wait(lock, [this] {
            return !m_queue.empty() || m_finished || m_reader_state != ReaderState::OPEN;
        });
        if (m_reader_state != ReaderState::OPEN) {
            return std::nullopt;
        }
        return pop(lock);
    }

    bool truncated() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_truncated;
    }

    void detach() {
        close_read(ReaderState::DETACHED);
    }

    void abort() {
        close_read(ReaderState::ABORTED);
    }

private:
    void close_read(ReaderState state) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_reader_state != ReaderState::ABORTED) {
                m_reader_state = state;
            }
            m_queue.clear();
        }
        m_cv.notify_all();
    }

    /// @brief Pop the front step under an already-held lock. nullopt when the queue is empty.
    std::optional<ov::AnyMap> pop(const std::unique_lock<std::mutex>&) {
        if (m_queue.empty()) {
            return std::nullopt;
        }
        ov::AnyMap data = std::move(m_queue.front());
        m_queue.pop_front();
        return data;
    }

    std::deque<ov::AnyMap> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_finished = false;
    bool m_truncated = false;
    ReaderState m_reader_state = ReaderState::OPEN;
};

OmniChannel::OmniChannel() : m_impl(std::make_unique<Impl>()) {}

OmniChannel::~OmniChannel() = default;

StreamingStatus OmniChannel::write(const ov::AnyMap& data) {
    return m_impl->write(data);
}

void OmniChannel::end() {
    m_impl->end();
}

void OmniChannel::abandon() {
    m_impl->abandon();
}

std::optional<ov::AnyMap> OmniChannel::read() {
    return m_impl->read();
}

bool OmniChannel::truncated() const {
    return m_impl->truncated();
}

void OmniChannel::detach() {
    m_impl->detach();
}

void OmniChannel::abort() {
    m_impl->abort();
}

}  // namespace genai
}  // namespace ov
