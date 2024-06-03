// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

class TextCallbackStreamer: public StreamerBase {
public:
    bool put(int64_t token) override;
    void end() override;

    TextCallbackStreamer(const Tokenizer& tokenizer, std::function<bool(std::string)> callback);
   
    std::function<bool(std::string)> on_finalized_subword_callback = [](std::string words)->bool { return false; };
    int64_t m_eos_token;
private:
    Tokenizer m_tokenizer;
    std::vector<int64_t> m_tokens_cache;
    size_t print_len = 0;
};

}  // namespace genai
}  // namespace ov
