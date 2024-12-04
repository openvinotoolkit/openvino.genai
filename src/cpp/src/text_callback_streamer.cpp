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
    
    // In some cases adding the next token can shorten the text, 
    // e.g. when apostrophe removing regex had worked after adding new tokens.
    // Need to hold on before flushing if apostrophe is the last symbol.
    if (text.size() > 1 && text.back() == '\'') {
        return on_finalized_subword_callback(res.str());
    }
    
    if (!text.empty() && '\n' == text.back() && text.size() > print_len) {
        // Flush the cache after the new line symbol
        res << std::string_view{text.data() + print_len, text.size() - print_len};
        m_tokens_cache.clear();
        print_len = 0;
        return on_finalized_subword_callback(res.str());
    }

    constexpr char replacement[] = "\xef\xbf\xbd";  // MSVC with /utf-8 fails to compile � directly with newline in string literal error.
    if (text.size() >= 3 && text.compare(text.size() - 3, 3, replacement) == 0) {
        // Don't print incomplete text
        return on_finalized_subword_callback(res.str());
    } else if (text.size() > print_len) {
        // It is possible to have a shorter text after adding new token.
        // Print to output only if text length is increaesed.
        res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    return on_finalized_subword_callback(res.str());
}

void TextCallbackStreamer::end() {
    std::stringstream res;
    std::string text = m_tokenizer.decode(m_tokens_cache);
    if (text.size() <= print_len)
        return ;
    res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
    m_tokens_cache.clear();
    print_len = 0;
    on_finalized_subword_callback(res.str());
    return;
}

}  // namespace genai
}  // namespace ov
