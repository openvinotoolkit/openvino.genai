// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "streamer.hpp"

#include "openvino/genai/text_streamer.hpp"

namespace ov {
namespace genai {

StreamingStatus ChunkTextCallbackStreamer::write(int64_t token) {
    return ov::genai::TextStreamer::write(token);
}

StreamingStatus ChunkTextCallbackStreamer::write_chunk(std::vector<int64_t> tokens) {
    if (tokens.empty()) {
        return ov::genai::StreamingStatus::STOP;
    }

    if (tokens.size() > 1) {
        m_tokens_cache.insert(m_tokens_cache.end(), tokens.begin(), tokens.end() - 1);
    }

    return ov::genai::TextStreamer::write(tokens.back());
}

void ChunkTextCallbackStreamer::end() {
    ov::genai::TextStreamer::end();
}

}  // namespace genai
}  // namespace ov
