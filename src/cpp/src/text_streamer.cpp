// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text_streamer.hpp"

namespace {
bool is_incomplete(std::string& text) {
    // MSVC with /utf-8 fails to compile � directly with newline in string literal error.
    constexpr char replacement[] = "\xef\xbf\xbd";
    return text.size() >= 3 && text.compare(text.size() - 3, 3, replacement) == 0;
}

constexpr size_t delay_n_tokens = 3;

}  // namespace

namespace ov {
namespace genai {

TextStreamer::TextStreamer(const Tokenizer& tokenizer,
                           std::function<ov::genai::CallbackTypeVariant(std::string)> callback,
                           const ov::AnyMap& detokenization_params) {
    m_tokenizer = tokenizer;
    m_subword_callback = std::move(callback);
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

        auto res_status = run_callback_if_needed(res.str());
        m_tokens_cache.clear();
        m_decoded_lengths.clear();
        m_printed_len = 0;
        return res_status;
    }

    if (is_incomplete(text)) {
        m_decoded_lengths[m_decoded_lengths.size() - 1] = -1;
        // Don't print incomplete text
        return run_callback_if_needed(res.str());
    }

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
        // Print to output only if text length is increased.
        res << std::string_view{text.data() + m_printed_len, print_until - m_printed_len} << std::flush;
    }
    
    auto status = run_callback_if_needed(res.str());
    
    if (print_until > -1 && print_until > m_printed_len) {
        m_printed_len = print_until;
    }
    return status;
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
    std::string text = m_tokenizer.decode(m_tokens_cache, m_additional_detokenization_params);
    auto clear_cache = [this]() {
        m_tokens_cache.clear();
        m_decoded_lengths.clear();
        m_printed_len = 0;
    };
    // Call the callback before clearing the cache, so that TextParserStreamer can still see the tokens of the last chunk.
    if (text.size() > m_printed_len) {
        try {
            m_subword_callback(text.substr(m_printed_len));
        } catch (...) {
            clear_cache();
            throw;
        }
    }
    clear_cache();
}

StreamerBase::~StreamerBase() = default;

// Is used to hide internal states of TextParserStreamer
class TextParserStreamer::TextParserStreamerImpl {
public:

std::vector<std::shared_ptr<IncrementalParser>> m_parsers;
JsonContainer m_parsed_message;
// Number of tokens at the start of m_tokens_cache that were already passed to the parsers.
size_t m_flushed_tokens = 0;
// Set while end() flushes the rest of the cache.
bool m_ending = false;

TextParserStreamerImpl(std::vector<std::shared_ptr<IncrementalParser>> parsers) : m_parsers{parsers} {}

};

TextParserStreamer::TextParserStreamer(const Tokenizer& tokenizer, std::vector<std::shared_ptr<IncrementalParser>> parsers) 
    : TextStreamer(tokenizer, [this](std::string s) -> CallbackTypeVariant {
                return this->write(s);
    }), m_pimpl{std::make_unique<TextParserStreamerImpl>(parsers)} {
        m_pimpl->m_parsed_message["content"] = "";
    }

CallbackTypeVariant TextParserStreamer::write(std::string delta_text) {
    // When 'write' is called with string, it means new chunk of tokens is decoded into text

    // prompt          = "I was waiting for the bus.\n"
    // tokens          = [2,2,  3,      45, 67, 89,4,2]
    // decoded_lengths = [1,5,  13,     17, 21, 25,26,27]
    // let printed_len = 13 (after "I was waiting"), so m_flushed_tokens = 3
    // then delta_text = "for the bus.\n"
    // delta_tokens = [45, 67, 89,4,2]
    // delta_tokens = m_tokens_cache[3..end]
    //
    // The chunk boundaries are tracked by counting the tokens already passed to the parsers. m_decoded_lengths
    // can't be searched for them: it holds -1 for incomplete UTF-8 and -2 for positions which were not decoded
    // after a multi-token write, and tokens which add no text have the same length as the token before them.
    size_t& flushed = m_pimpl->m_flushed_tokens;
    // TextStreamer prints the rest of the cache and clears it afterwards in end() and when the whole decoded text
    // ends with a new line. A multi-token write can also have a new line in the middle of the text; then the
    // delayed part of delta_text may end with '\n' while the cache is kept.
    const bool flush_all = m_pimpl->m_ending ||
                           (!delta_text.empty() && delta_text.back() == '\n' && !m_decoded_lengths.empty() &&
                            m_decoded_lengths.back() == static_cast<int64_t>(m_printed_len + delta_text.size()));
    size_t last = flushed;
    if (flush_all) {
        last = m_tokens_cache.size();
    } else if (m_decoded_lengths.size() >= delay_n_tokens) {
        // delta_text ends where the text of the tokens up to print_position ends.
        const size_t print_position = m_decoded_lengths.size() - delay_n_tokens;
        const auto print_until = m_decoded_lengths[print_position];
        last = print_position + 1;
        // Tokens right after it which add no text (e.g. special tokens skipped by the detokenizer) go with this chunk.
        // After a multi-token write their lengths may not be decoded yet.
        while (last < m_decoded_lengths.size()) {
            compute_decoded_length_for_position(last);
            if (m_decoded_lengths[last] != print_until) {
                break;
            }
            ++last;
        }
    }

    auto flushed_tokens = std::vector<int64_t>();
    if (last > flushed) {
        flushed_tokens.assign(m_tokens_cache.begin() + flushed, m_tokens_cache.begin() + last);
        flushed = last;
    }
    if (flush_all) {
        flushed = 0;
    }

    // Every time we start to cycle through iterative parsers we create a new delta_message.
    // Parsers should neither delete fields nor rewrite existing non-string data; they may only add fields
    // to delta_message or set string fields whose values will be concatenated with the accumulated message.
    StreamingStatus parser_last_status = StreamingStatus::RUNNING;

    JsonContainer delta_message;
    // Iterate over all parsers and apply them to the message
    for (auto& parser: m_pimpl->m_parsers) {
        delta_text = parser->parse(delta_message, delta_text, flushed_tokens);
        parser_last_status = parser->get_status();
        // If there is any stop status from parser we will stop streaming and this signal
        // has priority over streaming status from write method of TextStreamer.
        if (parser_last_status != StreamingStatus::RUNNING) {
            // Reset parser status to RUNNING should be done in reset() method implementation.
            break;
        }
        // Message can be modified inside parser, if parser for example extracted tool calling from message content
    }
    delta_message["content"] = delta_text;
    
    m_pimpl->m_parsed_message.concatenate(delta_message);
    auto stream_status = write(delta_message);
    if (parser_last_status != StreamingStatus::RUNNING) {
        // stopping statuses from parser have priority over status from write
        return parser_last_status;
    }
    return stream_status;
}

JsonContainer TextParserStreamer::get_parsed_message() const {
    return m_pimpl->m_parsed_message;
}

void TextParserStreamer::end() {
    m_pimpl->m_ending = true;
    try {
        TextStreamer::end();
    } catch (...) {
        m_pimpl->m_ending = false;
        m_pimpl->m_flushed_tokens = 0;
        throw;
    }
    m_pimpl->m_ending = false;
    m_pimpl->m_flushed_tokens = 0;
}

void TextParserStreamer::reset() {
    m_tokens_cache.clear();
    m_decoded_lengths.clear();
    m_printed_len = 0;
    m_pimpl->m_flushed_tokens = 0;
    m_pimpl->m_ending = false;
    m_pimpl->m_parsed_message = JsonContainer();
    m_pimpl->m_parsed_message["content"] = "";
    for (auto& parser : m_pimpl->m_parsers) {
        parser->reset();
    }
}

TextParserStreamer::~TextParserStreamer() = default;

}  // namespace genai
}  // namespace ov
