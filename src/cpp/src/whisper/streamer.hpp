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
    ChunkToBaseStreamerAdapter(std::shared_ptr<ChunkStreamerBase> chunk_streamer) : m_chunk_streamer{chunk_streamer} {}

    StreamingStatus write(const std::vector<int64_t>& tokens) override {
        return m_chunk_streamer->put_chunk(tokens) ? StreamingStatus::STOP : StreamingStatus::RUNNING;
    }

    StreamingStatus write(int64_t token) override {
        return m_chunk_streamer->put(token) ? StreamingStatus::STOP : StreamingStatus::RUNNING;
    }

    void end() override {
        return m_chunk_streamer->end();
    }

private:
    std::shared_ptr<ChunkStreamerBase> m_chunk_streamer;
};

}  // namespace genai
}  // namespace ov
