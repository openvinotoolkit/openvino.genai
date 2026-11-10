// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "sampling/sampler.hpp"
#include "whisper/models/decoder.hpp"

namespace ov::genai {

/**
 * Ensures decoder and sampler cleanup if a Whisper generation round fails.
 *
 * On success, the caller resets the decoder and dismisses the guard.
 * Otherwise, destruction waits for in-flight inference, resets the
 * decoder, and clears tracked sampler requests. Cleanup errors are suppressed.
 */
class DecoderRoundGuard {
public:
    DecoderRoundGuard(std::shared_ptr<WhisperDecoder> decoder, Sampler& sampler)
        : m_decoder(std::move(decoder)),
          m_sampler(sampler) {}

    DecoderRoundGuard(const DecoderRoundGuard&) = delete;
    DecoderRoundGuard& operator=(const DecoderRoundGuard&) = delete;

    void track_request(uint64_t request_id) {
        m_request_ids.push_back(request_id);
    }

    void dismiss() {
        clear_requests();
        m_armed = false;
    }

    ~DecoderRoundGuard() {
        if (!m_armed) {
            return;
        }
        try {
            m_decoder->wait();
        } catch (...) {
        }
        try {
            m_decoder->reset_state();
        } catch (...) {
        }
        try {
            clear_requests();
        } catch (...) {
        }
    }

private:
    void clear_requests() {
        for (const uint64_t request_id : m_request_ids) {
            m_sampler.clear_request_info(request_id);
        }
        m_request_ids.clear();
    }

    std::shared_ptr<WhisperDecoder> m_decoder;
    Sampler& m_sampler;
    std::vector<uint64_t> m_request_ids;
    bool m_armed = true;
};

}  // namespace ov::genai
