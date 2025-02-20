// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/text_streamer.hpp"

namespace ov {
namespace genai {

class ChunkTextCallbackStreamer : private TextStreamer, public ChunkStreamerBase {
public:
    StreamingStatus write(int64_t token) override;
    StreamingStatus write_chunk(std::vector<int64_t> tokens) override;
    void end() override;

    ChunkTextCallbackStreamer(const Tokenizer& tokenizer, std::function<ov::genai::CallbackTypeVariant(std::string)> callback)
        : TextStreamer(tokenizer, callback){};
};

}  // namespace genai
}  // namespace ov
