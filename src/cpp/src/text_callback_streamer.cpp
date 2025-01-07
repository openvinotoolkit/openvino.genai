// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_callback_streamer.hpp"

namespace ov {
namespace genai {

TextCallbackStreamer::TextCallbackStreamer(const Tokenizer& tokenizer, std::function<bool(std::string)> callback) {
    m_tokenizer = tokenizer;
    on_finalized_subword_callback = callback;
}

bool TextCallbackStreamer::put(int64_t token) {
    std::stringstream res;
    m_tokens_cache.push_back(token);
    std::string text = m_tokenizer.decode(m_tokens_cache);
    m_decoded_lengths.push_back(text.length());
    
    
    if (!text.empty() && '\n' == text.back() && text.size() > printed_len) {
        // Flush the cache after the new line symbol
        res << std::string_view{text.data() + printed_len, text.size() - printed_len};
        m_tokens_cache.clear();
        m_decoded_lengths.clear();
        printed_len = 0;
        return on_finalized_subword_callback(res.str());
    }

    // In some cases adding the next token can shorten the text, 
    // e.g. when apostrophe removing regex had worked after adding new tokens.
    // Printing several last tokens is delayed.
    constexpr size_t delay_n_tokens = 3;
    if (m_tokens_cache.size() < delay_n_tokens) {
        return on_finalized_subword_callback(res.str());
    }

    auto print_until = m_decoded_lengths[m_decoded_lengths.size() - delay_n_tokens];
    constexpr char replacement[] = "\xef\xbf\xbd";  // MSVC with /utf-8 fails to compile � directly with newline in string literal error.
    if (text.size() >= 3 && text.compare(text.size() - 3, 3, replacement) == 0) {
        // Don't print incomplete text
        return on_finalized_subword_callback(res.str());
    } else if (print_until > printed_len) {
        // It is possible to have a shorter text after adding new token.
        // Print to output only if text length is increaesed.
        res << std::string_view{text.data() + printed_len, print_until - printed_len} << std::flush;
        printed_len = print_until;
    }

    return on_finalized_subword_callback(res.str());
}

void TextCallbackStreamer::end() {
    std::stringstream res;
    std::string text = m_tokenizer.decode(m_tokens_cache);
    if (text.size() <= printed_len)
        return ;
    res << std::string_view{text.data() + printed_len, text.size() - printed_len} << std::flush;
    m_tokens_cache.clear();
    printed_len = 0;
    on_finalized_subword_callback(res.str());
    return;
}

}  // namespace genai
}  // namespace ov
