// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text_streamer.hpp"

namespace {
bool is_incomplete(std::string& text) {
    // MSVC with /utf-8 fails to compile � directly with newline in string literal error.
    constexpr char replacement[] = "\xef\xbf\xbd";
    return text.size() >= 3 && text.compare(text.size() - 3, 3, replacement) == 0;
}
}  // namespace

namespace ov {
namespace genai {

TextStreamer::TextStreamer(const Tokenizer& tokenizer,
                           std::function<ov::genai::CallbackTypeVariant(std::string)> callback,
                           const ov::AnyMap& detokenization_params) {
    m_tokenizer = tokenizer;
    m_subword_callback = callback;
    m_additional_detokenization_params = detokenization_params;
}

StreamingStatus TextStreamer::write(int64_t token) {
    std::stringstream res;
    m_tokens_cache.push_back(token);
    std::string text = m_tokenizer.decode(m_tokens_cache, m_additional_detokenization_params);
    m_decoded_lengths.push_back(text.length());

    if (!text.empty() && '\n' == text.back() && text.size() > m_printed_len) {
        // Flush the cache after the new line symbol
        res << std::string_view{text.data() + m_printed_len, text.size() - m_printed_len};
        m_tokens_cache.clear();
        m_decoded_lengths.clear();
        m_printed_len = 0;
        return run_callback_if_needed(res.str());
    }

    if (is_incomplete(text)) {
        m_decoded_lengths[m_decoded_lengths.size() - 1] = -1;
        // Don't print incomplete text
        return run_callback_if_needed(res.str());
    }
    constexpr size_t delay_n_tokens = 3;
    // In some cases adding the next token can shorten the text,
    // e.g. when apostrophe removing regex had worked after adding new tokens.
    // Printing several last tokens is delayed.
    if (m_decoded_lengths.size() < delay_n_tokens) {
        return run_callback_if_needed(res.str());
    }

    compute_decoded_length_for_position(m_decoded_lengths.size() - delay_n_tokens);

    auto print_until = m_decoded_lengths[m_decoded_lengths.size() - delay_n_tokens];

    if (print_until > -1 && print_until > m_printed_len) {
        // It is possible to have a shorter text after adding new token.
        // Print to output only if text length is increaesed.
        res << std::string_view{text.data() + m_printed_len, print_until - m_printed_len} << std::flush;
        m_printed_len = print_until;
    }

    return run_callback_if_needed(res.str());
}

void TextStreamer::compute_decoded_length_for_position(size_t cache_position) {
    // decode was performed for this position, skippping
    if (m_decoded_lengths[cache_position] != -2) {
        return;
    }

    auto cache_for_position = std::vector(m_tokens_cache.begin(), m_tokens_cache.begin() + cache_position + 1);
    std::string text_for_position = m_tokenizer.decode(cache_for_position, m_additional_detokenization_params);

    if (is_incomplete(text_for_position)) {
        m_decoded_lengths[cache_position] = -1;
    } else {
        m_decoded_lengths[cache_position] = text_for_position.size();
    }
};

StreamingStatus TextStreamer::write(const std::vector<int64_t>& tokens) {
    if (tokens.empty()) {
        return StreamingStatus::RUNNING;
    }

    if (tokens.size() > 1) {
        m_tokens_cache.insert(m_tokens_cache.end(), tokens.begin(), tokens.end() - 1);
        // -2 means no decode was done for this token position
        m_decoded_lengths.resize(m_decoded_lengths.size() + tokens.size() - 1, -2);
    }

    return ov::genai::TextStreamer::write(tokens.back());
}

StreamingStatus TextStreamer::set_streaming_status(CallbackTypeVariant callback_status) {
    if (auto res = std::get_if<StreamingStatus>(&callback_status))
        return *res;
    else
        return std::get<bool>(callback_status) ? StreamingStatus::STOP : StreamingStatus::RUNNING;
}

StreamingStatus TextStreamer::run_callback_if_needed(const std::string& text) {
    if (text.empty()) {
        return StreamingStatus::RUNNING;
    } else {
        return set_streaming_status(m_subword_callback(text));
    }
}

void TextStreamer::end() {
    std::stringstream res;
    std::string text = m_tokenizer.decode(m_tokens_cache, m_additional_detokenization_params);
    if (text.size() <= m_printed_len)
        return;
    res << std::string_view{text.data() + m_printed_len, text.size() - m_printed_len} << std::flush;
    m_tokens_cache.clear();
    m_decoded_lengths.clear();
    m_printed_len = 0;
    m_subword_callback(res.str());
    return;
}

ov::genai::StreamerBase::~StreamerBase() = default;

}  // namespace genai
}  // namespace ov