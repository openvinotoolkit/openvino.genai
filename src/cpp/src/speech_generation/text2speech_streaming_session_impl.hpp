// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <optional>
#include <string>

#include "openvino/genai/speech_generation/text2speech_streaming_session.hpp"

namespace ov::genai {

/// Occupies one of the concurrent streaming sessions a Text2SpeechPipelineImpl allows.
class Text2SpeechStreamingSessionSlot {
public:
    Text2SpeechStreamingSessionSlot() = default;
    explicit Text2SpeechStreamingSessionSlot(std::atomic<size_t>* counter) : m_counter{counter} {}
    Text2SpeechStreamingSessionSlot(Text2SpeechStreamingSessionSlot&& other) noexcept : m_counter{other.m_counter} {
        other.m_counter = nullptr;
    }
    Text2SpeechStreamingSessionSlot& operator=(Text2SpeechStreamingSessionSlot&& other) noexcept {
        if (this != &other) {
            release();
            m_counter = other.m_counter;
            other.m_counter = nullptr;
        }
        return *this;
    }
    Text2SpeechStreamingSessionSlot(const Text2SpeechStreamingSessionSlot&) = delete;
    Text2SpeechStreamingSessionSlot& operator=(const Text2SpeechStreamingSessionSlot&) = delete;
    ~Text2SpeechStreamingSessionSlot() {
        release();
    }

private:
    void release() {
        if (m_counter) {
            m_counter->fetch_sub(1);
            m_counter = nullptr;
        }
    }
    std::atomic<size_t>* m_counter = nullptr;
};

/// Backend side of Text2SpeechStreamingSession: the push() / end() / read() contract documented there.
class Text2SpeechStreamingSessionImpl {
public:
    virtual ~Text2SpeechStreamingSessionImpl() = default;
    virtual void push(const std::string& text) = 0;
    virtual void end() = 0;
    virtual std::optional<Text2SpeechStreamingResult> read() = 0;
};

}  // namespace ov::genai
