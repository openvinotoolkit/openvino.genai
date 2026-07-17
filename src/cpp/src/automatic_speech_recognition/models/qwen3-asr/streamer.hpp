// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/genai/streamer_base.hpp"

namespace ov::genai {

/**
 * Qwen3ASRStreamer is a wrapper around a user-provided StreamerBase instance. It suppresses all tokens until the first
 * occurrence of a specific token (m_asr_text_token_id) is encountered.
 * It's required to handle Qwen3-ASR language autodetection prefix - "language English<asr_text>..."
 */
class Qwen3ASRStreamer final : public StreamerBase {
public:
    Qwen3ASRStreamer(std::shared_ptr<StreamerBase> streamer, int64_t asr_text_token_id, bool suppress_until_asr_text);

    StreamingStatus write(int64_t token) override;
    StreamingStatus write(const std::vector<int64_t>& tokens) override;
    void end() override;

private:
    std::shared_ptr<StreamerBase> m_streamer;
    int64_t m_asr_text_token_id;
    bool m_forwarding;
};

}  // namespace ov::genai
