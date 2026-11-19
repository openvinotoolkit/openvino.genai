// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/streaming_session.hpp"

namespace ov::genai {

class ASRPipelineImplBase;

/// Result of the legacy push_chunk()/finish() session interface. Used only by sessions not yet ported to process().
struct ASRPartialResult {
    std::string language;
    std::string committed_text;
    std::string new_committed_text;
    std::string partial_text;
};

/// Occupies one of the concurrent streaming sessions allowed by an ASRPipelineImplBase.
class ASRStreamingSessionSlot {
public:
    ASRStreamingSessionSlot() = default;
    explicit ASRStreamingSessionSlot(std::atomic<size_t>* counter) : m_counter{counter} {}
    ASRStreamingSessionSlot(ASRStreamingSessionSlot&& other) noexcept : m_counter{other.m_counter} {
        other.m_counter = nullptr;
    }
    ASRStreamingSessionSlot& operator=(ASRStreamingSessionSlot&& other) noexcept {
        if (this != &other) {
            release();
            m_counter = other.m_counter;
            other.m_counter = nullptr;
        }
        return *this;
    }
    ASRStreamingSessionSlot(const ASRStreamingSessionSlot&) = delete;
    ASRStreamingSessionSlot& operator=(const ASRStreamingSessionSlot&) = delete;
    ~ASRStreamingSessionSlot() {
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

/**
 * @brief Implements the push()/end()/read() contract of ASRStreamingSession.
 *
 * Splits buffered audio into decode passes of exactly chunk_size samples. After end() the remaining audio forms the
 * last pass. Subclasses implement process() for one pass. Sessions not yet ported to process() override the legacy
 * push_chunk()/finish() interface instead; the default process() forwards to it.
 */
class ASRStreamingSessionImpl {
public:
    /// Legacy constructor. The chunk size is taken from ASRStreamingConfig in attach().
    ASRStreamingSessionImpl() = default;
    explicit ASRStreamingSessionImpl(size_t chunk_size_samples);
    virtual ~ASRStreamingSessionImpl() = default;

    /// Legacy interface: buffer audio and decode once a chunk is accumulated.
    virtual std::optional<ASRPartialResult> push_chunk(const std::vector<float>& pcm16k);
    /// Legacy interface: decode the remaining audio and commit the unstable tail.
    virtual ASRPartialResult finish();

    void push(const std::vector<float>& pcm16k);
    void end();
    std::optional<ASRStreamingResult> read();

    /// Keep the pipeline implementation and the session slot alive for the lifetime of the session.
    void attach(std::shared_ptr<ASRPipelineImplBase> owner, ASRStreamingSessionSlot slot, size_t chunk_size_samples);

protected:
    /// Run one decode pass. audio holds chunk_size samples, or the remaining samples (possibly none) when is_last.
    virtual ASRStreamingResult process(const std::vector<float>& audio, bool is_last);

private:
    size_t m_chunk_size_samples = 0;

    std::mutex m_input_mutex;
    std::vector<float> m_buffer;
    bool m_ended = false;

    std::mutex m_read_mutex;
    bool m_final_returned = false;

    // Declared last so that the slot is released and the owner destroyed after subclass members.
    std::shared_ptr<ASRPipelineImplBase> m_owner;
    ASRStreamingSessionSlot m_slot;
};

}  // namespace ov::genai
