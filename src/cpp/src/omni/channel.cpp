// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/channel.hpp"

#include <condition_variable>
#include <deque>
#include <iostream>  // TODO: temporary, for trace_write().
#include <mutex>
#include <utility>
#include <vector>

namespace ov {
namespace genai {

/// @brief Queue behind OmniChannel. Pimpl'd so <mutex> / <condition_variable> stay out of the
/// public header — a std::mutex member would otherwise cross the shared-library boundary.
class OmniChannel::Impl {
public:
    StreamingStatus write(const ov::AnyMap& data) {
        size_t depth = 0;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queue.push_back(data);
            depth = m_queue.size();
        }
        m_cv.notify_one();
        // TODO: temporary — drop once the talker reads the other end. Traces what the thinker
        // pushes so the bridge can be watched end-to-end before there is a consumer.
        trace_write(data, depth);
        return StreamingStatus::RUNNING;
    }

    void end() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_finished = true;
        }
        m_cv.notify_all();
    }

    std::optional<ov::AnyMap> read() {
        std::unique_lock<std::mutex> lock(m_mutex);
        // Queued steps win over m_finished, so end() never discards what was already written:
        // the reader keeps draining and only sees the end of the stream once the queue is empty.
        m_cv.wait(lock, [this] {
            return !m_queue.empty() || m_finished;
        });
        return pop(lock);
    }

private:
    // TODO: temporary — remove together with the trace_write() call in write().
    void trace_write(const ov::AnyMap& data, size_t queue_depth) {
        const auto tokens_it = data.find(omni_stream::tokens.name());
        const auto tokens = tokens_it == data.end() ? std::vector<int64_t>{}
                                                    : tokens_it->second.as<std::vector<int64_t>>();

        std::cerr << "[OmniChannel] write #" << m_written++ << ": " << tokens.size() << " token(s) [";
        for (size_t i = 0; i < tokens.size() && i < 8; ++i) {
            std::cerr << (i ? ", " : "") << tokens[i];
        }
        std::cerr << (tokens.size() > 8 ? ", ...]" : "]");

        const auto hidden_states_it = data.find(omni_stream::hidden_states.name());
        if (hidden_states_it == data.end()) {
            std::cerr << ", no hidden states";
        } else {
            const auto hidden_states = hidden_states_it->second.as<std::vector<ov::Tensor>>();
            std::cerr << ", " << hidden_states.size() << " hidden state(s)";
            if (!hidden_states.empty()) {
                std::cerr << " of shape " << hidden_states.front().get_shape();
            }
        }
        std::cerr << ", queue depth " << queue_depth << std::endl;
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
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_finished = false;
    size_t m_written = 0;  // TODO: temporary, only used by trace_write().
};

OmniChannel::OmniChannel() : m_impl(std::make_unique<Impl>()) {}

OmniChannel::~OmniChannel() = default;

StreamingStatus OmniChannel::write(const ov::AnyMap& data) {
    return m_impl->write(data);
}

void OmniChannel::end() {
    m_impl->end();
}

std::optional<ov::AnyMap> OmniChannel::read() {
    return m_impl->read();
}

}  // namespace genai
}  // namespace ov
