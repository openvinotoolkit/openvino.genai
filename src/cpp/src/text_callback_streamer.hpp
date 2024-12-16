// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

class TextCallbackStreamer: public StreamerBase {
public:
    bool put(int64_t token) override;
    void end() override;

    TextCallbackStreamer(const Tokenizer& tokenizer, std::function<bool(std::string)> callback, const std::set<std::string>& stop_strings = {});

    std::function<bool(std::string)> on_finalized_subword_callback = [](std::string words)->bool { return false; };

protected:
    Tokenizer m_tokenizer;
    std::vector<int64_t> m_tokens_cache;
    std::list<int64_t> m_tokens_cache_stop_string;
    size_t print_len = 0, m_max_stop_string_len = 0;
    std::set<std::string> m_stop_strings;
};

}  // namespace genai
}  // namespace ov
