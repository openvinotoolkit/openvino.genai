// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

class TextCallbackStreamer : public StreamerBase {
public:
    TextCallbackStreamer(const Tokenizer& tokenizer, std::function<bool(std::string)> callback);

    bool put(int64_t token) override;
    bool put(const std::vector<int64_t>& tokens) override;
    void end() override;

protected:
    Tokenizer m_tokenizer;
    std::vector<int64_t> m_tokens_cache;
    std::vector<int64_t> m_decoded_lengths;
    size_t m_printed_len = 0;

private:
    std::function<bool(std::string)> m_on_finalized_subword_callback = [](std::string words) -> bool {
        return false;
    };
};

}  // namespace genai
}  // namespace ov