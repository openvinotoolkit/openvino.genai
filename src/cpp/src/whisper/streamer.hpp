// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {

class ChunkToBaseStreamerAdapter : public StreamerBase {
public:
    OPENVINO_SUPPRESS_DEPRECATED_START
    ChunkToBaseStreamerAdapter(std::shared_ptr<ChunkStreamerBase> chunk_streamer) : m_chunk_streamer{chunk_streamer} {}
    OPENVINO_SUPPRESS_DEPRECATED_END

    StreamingStatus write(const std::vector<int64_t>& tokens) override {
        return m_chunk_streamer->put_chunk(tokens) ? StreamingStatus::STOP : StreamingStatus::RUNNING;
    }

    StreamingStatus write(int64_t token) override {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return m_chunk_streamer->put(token) ? StreamingStatus::STOP : StreamingStatus::RUNNING;
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    void end() override {
        return m_chunk_streamer->end();
    }

private:
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::shared_ptr<ChunkStreamerBase> m_chunk_streamer;
    OPENVINO_SUPPRESS_DEPRECATED_END
};

}  // namespace genai
}  // namespace ov
