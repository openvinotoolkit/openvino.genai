// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "omni/channel.hpp"

#include <utility>

namespace ov::genai {

StreamingStatus OmniChannel::write(const ov::AnyMap& data) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_reader_state != ReaderState::OPEN) {
            return m_reader_state == ReaderState::FAILED ? StreamingStatus::STOP : StreamingStatus::RUNNING;
        }
        m_queue.push_back(data);
    }
    m_cv.notify_one();
    return StreamingStatus::RUNNING;
}

void OmniChannel::end() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_finished = true;
    }
    m_cv.notify_all();
}

void OmniChannel::abandon() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_finished = true;
        m_truncated = true;
        m_queue.clear();
    }
    m_cv.notify_all();
}

std::optional<ov::AnyMap> OmniChannel::read() {
    std::unique_lock<std::mutex> lock(m_mutex);
    // Queued steps win over m_finished, so end() never discards what was already written.
    m_cv.wait(lock, [this] {
        return !m_queue.empty() || m_finished || m_reader_state != ReaderState::OPEN;
    });
    if (m_reader_state != ReaderState::OPEN || m_queue.empty()) {
        return std::nullopt;
    }
    ov::AnyMap data = std::move(m_queue.front());
    m_queue.pop_front();
    return data;
}

bool OmniChannel::truncated() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_truncated;
}

void OmniChannel::finish_reading() {
    close_read(ReaderState::FINISHED);
}

void OmniChannel::fail_reading() {
    close_read(ReaderState::FAILED);
}

void OmniChannel::close_read(ReaderState state) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_reader_state != ReaderState::FAILED) {
            m_reader_state = state;
        }
        m_queue.clear();
    }
    m_cv.notify_all();
}

}  // namespace ov::genai
