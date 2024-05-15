// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {

class TextCallbackStreamer: public StreamerBase {
public:
    void put(int64_t token) override;
    void end() override;

    TextCallbackStreamer(const Tokenizer& tokenizer, std::function<void (std::string)> callback, bool print_eos_token = false);
    TextCallbackStreamer(const Tokenizer& tokenizer, bool print_eos_token = false);
    TextCallbackStreamer() = default;
    ~TextCallbackStreamer() = default;

    void set_tokenizer(Tokenizer tokenizer);
    void set_callback(std::function<void (std::string)> callback);
    void set_callback();
    
    std::function<void (std::string)> on_decoded_text_callback = [](std::string words){};
    bool m_enabled = false;
    int64_t m_eos_token;
private:
    bool m_print_eos_token = false;
    Tokenizer m_tokenizer;
    std::vector<int64_t> m_tokens_cache;
    size_t print_len = 0;
    void on_finalized_text(const std::string& subword);
};

} // namespace ov
