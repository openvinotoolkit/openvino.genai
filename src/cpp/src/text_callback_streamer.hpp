// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

using CallbackTypeVariant = std::variant<bool, StreamingStatus, std::monostate>;

class TextCallbackStreamer: public StreamerBase {
    StreamingStatus set_streaming_status(CallbackTypeVariant callback_status);

    std::function<CallbackTypeVariant(std::string)> on_finalized_subword_callback = [](std::string words)->bool { return false; };

public:
    StreamingStatus write(int64_t token) override;

    void end() override;

    TextCallbackStreamer(const Tokenizer& tokenizer, std::function<CallbackTypeVariant(std::string)> callback);

protected:
    Tokenizer m_tokenizer;
    std::vector<int64_t> m_tokens_cache;
    std::vector<int64_t> m_decoded_lengths;
    size_t m_printed_len = 0;
};

}  // namespace genai
}  // namespace ov