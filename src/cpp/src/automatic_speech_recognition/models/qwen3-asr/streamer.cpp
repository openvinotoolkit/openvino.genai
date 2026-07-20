// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "streamer.hpp"

#include <algorithm>
#include <utility>

namespace ov::genai {

Qwen3ASRStreamer::Qwen3ASRStreamer(std::shared_ptr<StreamerBase> streamer,
                                   int64_t asr_text_token_id,
                                   bool suppress_until_asr_text)
    : m_streamer{std::move(streamer)},
      m_asr_text_token_id{asr_text_token_id},
      m_forwarding{!suppress_until_asr_text} {}

StreamingStatus Qwen3ASRStreamer::write(int64_t token) {
    if (!m_streamer) {
        return StreamingStatus::RUNNING;
    }

    if (!m_forwarding) {
        if (token == m_asr_text_token_id) {
            m_forwarding = true;
        }
        return StreamingStatus::RUNNING;
    }

    return m_streamer->write(token);
}

StreamingStatus Qwen3ASRStreamer::write(const std::vector<int64_t>& tokens) {
    if (!m_streamer || tokens.empty()) {
        return StreamingStatus::RUNNING;
    }

    if (m_forwarding) {
        return m_streamer->write(tokens);
    }

    const auto asr_text_it = std::find(tokens.begin(), tokens.end(), m_asr_text_token_id);
    if (asr_text_it == tokens.end()) {
        return StreamingStatus::RUNNING;
    }

    m_forwarding = true;
    if (asr_text_it + 1 == tokens.end()) {
        return StreamingStatus::RUNNING;
    }

    const std::vector<int64_t> transcript_tokens(asr_text_it + 1, tokens.end());
    return m_streamer->write(transcript_tokens);
}

void Qwen3ASRStreamer::end() {
    if (m_streamer) {
        m_streamer->end();
    }
}

}  // namespace ov::genai
