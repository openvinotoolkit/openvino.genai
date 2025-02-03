// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <queue>
#include <thread>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "text_callback_streamer.hpp"

namespace ov {
namespace genai {

class ChunkToBaseStreamerAdapter : public StreamerBase {
public:
    ChunkToBaseStreamerAdapter(std::shared_ptr<ChunkStreamerBase> chunk_streamer) : m_chunk_streamer{chunk_streamer} {}

    bool put(const std::vector<int64_t>& tokens) override {
        return m_chunk_streamer->put_chunk(tokens);
    }

    bool put(int64_t token) override {
        return m_chunk_streamer->put(token);
    }

    void end() override {
        return m_chunk_streamer->end();
    }

private:
    std::shared_ptr<ChunkStreamerBase> m_chunk_streamer;
};

}  // namespace genai
}  // namespace ov
