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
    
    if (!text.empty() && '\n' == text.back() && text.size() > m_printed_len) {
        // Flush the cache after the new line symbol
        res << std::string_view{text.data() + m_printed_len, text.size() - m_printed_len};
        m_tokens_cache.clear();
        m_decoded_lengths.clear();
        m_printed_len = 0;
        return on_finalized_subword_callback(res.str());
    }

    constexpr size_t delay_n_tokens = 3;
    // In some cases adding the next token can shorten the text, 
    // e.g. when apostrophe removing regex had worked after adding new tokens.
    // Printing several last tokens is delayed.
    if (m_decoded_lengths.size() < delay_n_tokens) {
        return on_finalized_subword_callback(res.str());
    }
    constexpr char replacement[] = "\xef\xbf\xbd";  // MSVC with /utf-8 fails to compile � directly with newline in string literal error.
    if (text.size() >= 3 && text.compare(text.size() - 3, 3, replacement) == 0) {
        m_decoded_lengths[m_decoded_lengths.size() - 1] = -1;
        // Don't print incomplete text
        return on_finalized_subword_callback(res.str());
    }
    auto print_until = m_decoded_lengths[m_decoded_lengths.size() - delay_n_tokens];
    if (print_until != -1 && print_until > m_printed_len) {
        // It is possible to have a shorter text after adding new token.
        // Print to output only if text length is increaesed.
        res << std::string_view{text.data() + m_printed_len, print_until - m_printed_len} << std::flush;
        m_printed_len = print_until;
    }

    return on_finalized_subword_callback(res.str());
}

void TextCallbackStreamer::end() {
    std::stringstream res;
    std::string text = m_tokenizer.decode(m_tokens_cache);
    if (text.size() <= m_printed_len)
        return;
    res << std::string_view{text.data() + m_printed_len, text.size() - m_printed_len} << std::flush;
    m_tokens_cache.clear();
    m_decoded_lengths.clear();
    m_printed_len = 0;
    on_finalized_subword_callback(res.str());
    return;
}

ov::genai::StreamerBase::~StreamerBase() = default;

}  // namespace genai
}  // namespace ov