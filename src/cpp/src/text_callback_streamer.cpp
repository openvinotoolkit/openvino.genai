// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_callback_streamer.hpp"

namespace ov {
namespace genai {

std::vector<int64_t> encode_and_process_stop_string(const std::string& stop_string, ov::genai::Tokenizer& tokenizer) {
    // encode stop_string
    ov::Tensor ov_encoded_stop_string = tokenizer.encode(stop_string).input_ids;
    size_t tensor_size = ov_encoded_stop_string.get_size();
    std::vector<int64_t> source_encoded_stop_string(tensor_size), encoded_stop_string;
    std::copy_n(ov_encoded_stop_string.data<int64_t>(), tensor_size, source_encoded_stop_string.begin());
    // remove special symbols
    for (const auto& token_id : source_encoded_stop_string) {
        if (token_id != tokenizer.get_bos_token_id() &&
            token_id != tokenizer.get_eos_token_id() &&
            token_id != tokenizer.get_pad_token_id()) {
            encoded_stop_string.push_back(token_id);
        }
    }
    return encoded_stop_string;
}

TextCallbackStreamer::TextCallbackStreamer(const Tokenizer& tokenizer, std::function<bool(std::string)> callback, const std::set<std::string>& stop_strings) {
    m_tokenizer = tokenizer;
    on_finalized_subword_callback = callback;
    for (const auto& stop_string : stop_strings) {
        auto encoded_stop_string = encode_and_process_stop_string(stop_string, m_tokenizer);
        m_max_stop_string_len = std::max(encoded_stop_string.size(), m_max_stop_string_len);
        m_stop_strings.insert(stop_string);
    }
}

bool TextCallbackStreamer::put(int64_t token) {
    std::stringstream res;
    m_tokens_cache_stop_string.push_back(token);
    if (m_tokens_cache_stop_string.size() > m_max_stop_string_len || token == m_tokenizer.get_eos_token_id()) {
        std::vector<int64_t> buffer(m_tokens_cache_stop_string.begin(), m_tokens_cache_stop_string.end());
        std::string text = m_tokenizer.decode(buffer);
        std::string activated_stop_string = "";
        for (const auto& stop_string : m_stop_strings) {
            if (text.find(stop_string) != std::string::npos) {
                activated_stop_string = stop_string;
                break;
            }
        }
        
        
        if (activated_stop_string.empty() && token != m_tokenizer.get_eos_token_id()) {
            m_tokens_cache.push_back(m_tokens_cache_stop_string.front());
            m_tokens_cache_stop_string.pop_front();
        } else {
            m_tokens_cache.insert(m_tokens_cache.end(), m_tokens_cache_stop_string.begin(), m_tokens_cache_stop_string.end());
            m_tokens_cache_stop_string.clear();
        }

        text = m_tokenizer.decode(m_tokens_cache);
        if (!activated_stop_string.empty()) {
            auto pos = text.find(activated_stop_string);
            if (pos != std::string::npos) {
                text.replace(pos, activated_stop_string.length(), "");
            }
            m_tokens_cache.clear();
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
        } else {
            // It is possible to have a shorter text after adding new token.
            // Print to output only if text length is increaesed.
            res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
            print_len = text.size();
        }
    }

    return on_finalized_subword_callback(res.str());
}

void TextCallbackStreamer::end() {
    std::stringstream res;
    std::vector<int64_t> buffer(m_tokens_cache.begin(), m_tokens_cache.end());
    std::string text = m_tokenizer.decode(buffer);
    if (text.size() <= print_len)
        return ;
    res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
    m_tokens_cache.clear();
    print_len = 0;
    on_finalized_subword_callback(res.str());
    return;
}

ov::genai::StreamerBase::~StreamerBase() = default;

}  // namespace genai
}  // namespace ov
