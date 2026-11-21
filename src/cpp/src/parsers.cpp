// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/parsers.hpp"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>
#include <cctype>
#include <stdexcept>

namespace ov::genai {

IncrementalParser::IncrementalParser() = default;
IncrementalParser::~IncrementalParser() = default;

class ReasoningIncrementalParser::ReasoningParserImpl {
private:
    // Values initialized from constructor don't need default member initializer.
    bool m_expect_open_tag;
    bool m_keep_original_content;
    std::string m_open_tag;
    std::string m_close_tag;
    // Values with default member initializers are reset on each reset() call.
    bool m_first_run = true;
    bool m_think_tag_opened = false;
    std::string m_text_cache = "";
    bool m_deactivated = false;

    /**
     * @brief Find the longest suffix of text that is a prefix of the close tag.
     * 
     * This is used to detect if the close tag is split across multiple chunks.
     * For example, if text ends with "</th" and m_close_tag is "</think>", 
     * this returns 4 (the length of "</th").
     * 
     * @param text The text to check (using string_view for efficient read-only access)
     * @return The number of characters at the end of text that match the start of m_close_tag
     */
    size_t find_close_tag_prefix_length(std::string_view text) const {
        const size_t max_check = std::min(text.size(), m_close_tag.size());
        size_t longest_match = 0;
        for (size_t i = 1; i <= max_check; ++i) {
            // Compare the last i characters of text with the first i characters of m_close_tag
            if (text.compare(text.size() - i, i, m_close_tag, 0, i) == 0) {
                longest_match = i;
            }
        }
        return longest_match;
    }

    /**
     * @brief Handle the case where both open and close tags are found in the same chunk.
     */
    void handle_complete_reasoning(JsonContainer& message, std::string_view txt_chunk,
                                   size_t open_idx, size_t close_idx, std::string& delta_text) {
        // Extract reasoning content between tags
        message["reasoning_content"] = std::string(txt_chunk.substr(open_idx + m_open_tag.size(), close_idx - (open_idx + m_open_tag.size())));
        message["content"] = std::string(txt_chunk.substr(close_idx + m_close_tag.size()));
        
        if (!m_keep_original_content) {
            delta_text = std::string(txt_chunk.substr(close_idx + m_close_tag.size()));
        }
        
        m_think_tag_opened = false;
        m_deactivated = true;
        m_text_cache.clear();
    }

    /**
     * @brief Handle the case where only the open tag is found.
     */
    void handle_open_tag(JsonContainer& delta_message, std::string_view txt_chunk, size_t open_idx, std::string& delta_text) {
        // Start accumulating reasoning content
        delta_message["reasoning_content"] = std::string(txt_chunk.substr(open_idx + m_open_tag.size()));
        
        if (!m_keep_original_content) {
            delta_text.clear();
        } else {
            delta_text = txt_chunk;
        }

        m_think_tag_opened = true;
        m_text_cache.clear();
    }

    /**
     * @brief Handle the case where the close tag is found.
     */
    void handle_close_tag(JsonContainer& delta_message, std::string_view txt_chunk, size_t close_idx, std::string& delta_text) {
        // Append text before close tag to reasoning content
        delta_message["reasoning_content"] = std::move(std::string(txt_chunk.substr(0, close_idx)));
        auto content = std::string(txt_chunk.substr(close_idx + m_close_tag.size()));
        delta_message["content"] = content;
        
        if (!m_keep_original_content) {
            // Despite the fact that we put txt_chunk to delta_text it's correct.
            // Since txt_chunk contains some cached parts from the previous calls that were not yet processed yet
            // and we kept them in cache until we decide what to do with them. Here we definitely know that that cached parts
            // belonged to reasoning_content so we can discard them.
            delta_text = std::move(content);
        } else {
            delta_text = txt_chunk;
        }

        m_text_cache.clear();
        m_think_tag_opened = false;
        m_deactivated = true;
    }

    /**
     * @brief Handle accumulating text while inside reasoning tags.
     */
    void handle_inside_reasoning(JsonContainer& delta_message, std::string_view txt_chunk, std::string& delta_text) {
        // Find if the end of txt_chunk might be the start of a close tag
        const size_t num_chars_to_keep = find_close_tag_prefix_length(txt_chunk);
        
        std::string reason_str;
        if (num_chars_to_keep > 0) {
            // Keep potential partial close tag in cache
            m_text_cache = std::string(txt_chunk.substr(txt_chunk.size() - num_chars_to_keep));
            reason_str = txt_chunk.substr(0, txt_chunk.size() - num_chars_to_keep);
        } else {
            // No partial close tag, accumulate all text
            reason_str = txt_chunk;
            m_text_cache.clear();
        }
        delta_message["reasoning_content"] = std::move(reason_str);
        if (m_keep_original_content) {
            delta_text = std::string(txt_chunk.substr(0, txt_chunk.size() - num_chars_to_keep));
        } else {
            delta_text.clear();
        }
    }

public:
    ReasoningParserImpl() = default;
    
    ReasoningParserImpl(bool expect_open_tag,
                    bool keep_original_content,
                    const std::string& open_tag, 
                    const std::string& close_tag)
        : m_expect_open_tag(expect_open_tag),
          m_keep_original_content(keep_original_content),
          m_open_tag(open_tag),
          m_close_tag(close_tag) {}

    std::string parse(
        JsonContainer&  delta_message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens
    ) {
        if (m_deactivated) {
            return delta_text;
        }
        if (!m_expect_open_tag && m_first_run) {
            m_think_tag_opened = true;
        }
        m_first_run = false;

        
        std::string txt_chunk = m_text_cache + delta_text;

        // Cache find() results to avoid redundant searches
        const auto open_idx = txt_chunk.find(m_open_tag);
        const auto close_idx = txt_chunk.find(m_close_tag);

        if (!m_think_tag_opened && open_idx != std::string::npos && m_expect_open_tag) {
            // Check if close tag is also present after the open tag
            const auto close_idx_after_open = (close_idx != std::string::npos && close_idx > open_idx) 
                                               ? close_idx : std::string::npos;
            
            if (close_idx_after_open != std::string::npos) {
                handle_complete_reasoning(delta_message, txt_chunk, open_idx, close_idx_after_open, delta_text);
            } else {
                handle_open_tag(delta_message, txt_chunk, open_idx, delta_text);
            }
        } else if (m_think_tag_opened && close_idx != std::string::npos) {
            handle_close_tag(delta_message, txt_chunk, close_idx, delta_text);
        } else if (m_think_tag_opened) {
            handle_inside_reasoning(delta_message, txt_chunk, delta_text);
        } else {
            // Think tag was not opened yet and not found in the current delta_text.
            // Accumulate text in the cache to detect if <think> is split between several delta_text pieces.
            m_text_cache += delta_text;
            // Clear delta_text while waiting for complete <think> tag detection in cache.
            // accumulating partial data in m_text_cache until the full <think> tag is detected.
            delta_text.clear();
        }
        return delta_text;
    }

    void reset() {
        m_first_run = true;
        m_think_tag_opened = false;
        m_text_cache = "";
        m_deactivated = false;
    }
};

ReasoningIncrementalParser::ReasoningIncrementalParser(bool expect_open_tag, bool keep_original_content, const std::string& open_tag, const std::string& close_tag) {
    m_impl = std::make_unique<ReasoningParserImpl>(expect_open_tag, keep_original_content, open_tag, close_tag);
}

ReasoningIncrementalParser::~ReasoningIncrementalParser() = default;

std::string ReasoningIncrementalParser::parse(
    JsonContainer& delta_message,
    std::string& delta_text,
    const std::optional<std::vector<int64_t>>& delta_tokens
) {
    return m_impl->parse(delta_message, delta_text, delta_tokens);
}

void ReasoningIncrementalParser::reset() {
    m_impl->reset();
}

namespace {

using Json = nlohmann::ordered_json;

// Parses a Python list of calls with literal keyword arguments: [f(a="x", b=[1, None]), g()]
// The text comes from a model, so anything else is not a tool call rather than an error.
// Checks bytes by hand because the <cctype> functions depend on the locale and split UTF-8.
class PythonicCallsReader {
public:
    PythonicCallsReader(const std::string& text, size_t pos) : m_text(text), m_pos(pos) {}

    std::optional<Json> read_calls() {
        Json calls = Json::array();
        if (!consume('[')) {
            return std::nullopt;
        }
        do {
            if (peek() == ']') {
                break;
            }
            auto call = read_call();
            if (!call) {
                return std::nullopt;
            }
            calls.push_back(std::move(*call));
        } while (consume(','));
        if (!consume(']') || calls.empty()) {
            return std::nullopt;
        }
        return calls;
    }

private:
    // Keeps deeply nested output from overflowing the stack
    static constexpr size_t MAX_DEPTH = 64;

    const std::string& m_text;
    size_t m_pos;
    size_t m_depth = 0;

    static bool is_digit(char c) {
        return c >= '0' && c <= '9';
    }

    static bool is_space(char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f';
    }

    static bool is_number_char(char c) {
        return is_digit(c) || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E';
    }

    // Bytes of non-ASCII characters are allowed, Python identifiers can be Unicode
    static bool is_identifier_start(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || static_cast<unsigned char>(c) >= 0x80;
    }

    char peek() {
        while (m_pos < m_text.size() && is_space(m_text[m_pos])) {
            ++m_pos;
        }
        return m_pos < m_text.size() ? m_text[m_pos] : '\0';
    }

    bool consume(char c) {
        if (peek() != c) {
            return false;
        }
        ++m_pos;
        return true;
    }

    std::string read_identifier() {
        const size_t start = m_pos;
        if (m_pos < m_text.size() && is_identifier_start(m_text[m_pos])) {
            while (m_pos < m_text.size() && (is_identifier_start(m_text[m_pos]) || is_digit(m_text[m_pos]))) {
                ++m_pos;
            }
        }
        return m_text.substr(start, m_pos - start);
    }

    // An identifier that Python allows as a name, so not a keyword like True or class
    std::string read_name() {
        static const std::set<std::string> keywords = {
            "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
            "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
            "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"};
        std::string name = read_identifier();
        return keywords.count(name) ? std::string{} : name;
    }

    // Function names can be dotted
    std::string read_function_name() {
        peek();
        std::string name = read_name();
        while (!name.empty() && m_pos < m_text.size() && m_text[m_pos] == '.') {
            ++m_pos;
            const std::string part = read_name();
            if (part.empty()) {
                return {};
            }
            name += '.' + part;
        }
        return name;
    }

    std::optional<Json> read_call() {
        std::string name = read_function_name();
        if (name.empty() || !consume('(')) {
            return std::nullopt;
        }
        Json arguments = Json::object();
        do {
            if (peek() == ')') {
                break;
            }
            peek();
            std::string key = read_name();
            if (key.empty() || arguments.contains(key) || !consume('=')) {
                return std::nullopt;
            }
            auto value = read_value();
            if (!value) {
                return std::nullopt;
            }
            arguments[key] = std::move(*value);
        } while (consume(','));
        if (!consume(')')) {
            return std::nullopt;
        }
        Json call = Json::object();
        call["name"] = std::move(name);
        call["arguments"] = std::move(arguments);
        return call;
    }

    std::optional<Json> read_value() {
        const char c = peek();
        if (c == '"' || c == '\'') {
            auto value = read_string();
            return value ? std::optional<Json>(std::move(*value)) : std::nullopt;
        }
        if (c == '[' || c == '{') {
            if (m_depth == MAX_DEPTH) {
                return std::nullopt;
            }
            ++m_depth;
            auto value = c == '[' ? read_list() : read_dict();
            --m_depth;
            return value;
        }
        if (c == '-' || is_digit(c)) {
            return read_number();
        }
        const std::string word = read_identifier();
        if (word == "True") {
            return Json(true);
        }
        if (word == "False") {
            return Json(false);
        }
        if (word == "None") {
            return Json(nullptr);
        }
        return std::nullopt;
    }

    static void append_utf8(std::string& out, uint32_t code_point) {
        if (code_point < 0x80) {
            out += static_cast<char>(code_point);
        } else if (code_point < 0x800) {
            out += static_cast<char>(0xC0 | (code_point >> 6));
            out += static_cast<char>(0x80 | (code_point & 0x3F));
        } else if (code_point < 0x10000) {
            out += static_cast<char>(0xE0 | (code_point >> 12));
            out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code_point & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (code_point >> 18));
            out += static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code_point & 0x3F));
        }
    }

    // Reads exactly digits hex characters of a \x, \u or \U escape into a Unicode code point
    bool read_hex_escape(size_t digits, std::string& out) {
        if (m_text.size() - m_pos < digits) {
            return false;
        }
        uint32_t code_point = 0;
        for (size_t i = 0; i < digits; ++i) {
            const char c = m_text[m_pos++];
            uint32_t digit = 0;
            if (is_digit(c)) {
                digit = c - '0';
            } else if (c >= 'a' && c <= 'f') {
                digit = c - 'a' + 10;
            } else if (c >= 'A' && c <= 'F') {
                digit = c - 'A' + 10;
            } else {
                return false;
            }
            code_point = (code_point << 4) | digit;
        }
        // Surrogates and values past Unicode have no UTF-8 form
        if (code_point > 0x10FFFF || (code_point >= 0xD800 && code_point <= 0xDFFF)) {
            return false;
        }
        append_utf8(out, code_point);
        return true;
    }

    bool read_escape(std::string& out) {
        if (m_pos == m_text.size()) {
            return false;
        }
        const char c = m_text[m_pos++];
        switch (c) {
        case '\n': return true;
        case '\r':
            if (m_pos < m_text.size() && m_text[m_pos] == '\n') {
                ++m_pos;
            }
            return true;
        case '\\': case '\'': case '"': out += c; return true;
        case 'a': out += '\a'; return true;
        case 'b': out += '\b'; return true;
        case 'f': out += '\f'; return true;
        case 'n': out += '\n'; return true;
        case 'r': out += '\r'; return true;
        case 't': out += '\t'; return true;
        case 'v': out += '\v'; return true;
        case 'x': return read_hex_escape(2, out);
        case 'u': return read_hex_escape(4, out);
        case 'U': return read_hex_escape(8, out);
        case 'N': return false;  // \N{name} needs the Unicode name table
        default: break;
        }
        if (c >= '0' && c <= '7') {
            uint32_t code_point = c - '0';
            for (int i = 0; i < 2 && m_pos < m_text.size() && m_text[m_pos] >= '0' && m_text[m_pos] <= '7'; ++i) {
                code_point = code_point * 8 + (m_text[m_pos++] - '0');
            }
            append_utf8(out, code_point);
            return true;
        }
        // Python keeps the backslash of an unknown escape
        out += '\\';
        out += c;
        return true;
    }

    std::optional<std::string> read_string() {
        const char quote = m_text[m_pos++];
        std::string value;
        while (m_pos < m_text.size() && m_text[m_pos] != quote) {
            const char c = m_text[m_pos++];
            if (c != '\\') {
                value += c;
            } else if (!read_escape(value)) {
                return std::nullopt;
            }
        }
        if (m_pos == m_text.size()) {
            return std::nullopt;
        }
        ++m_pos;
        return value;
    }

    // JSON's number syntax is the common part of Python's, and its parser ignores the locale
    std::optional<Json> read_number() {
        const size_t start = m_pos;
        while (m_pos < m_text.size() && is_number_char(m_text[m_pos])) {
            ++m_pos;
        }
        Json number = Json::parse(m_text.substr(start, m_pos - start), nullptr, /*allow_exceptions=*/false);
        return number.is_number() ? std::optional<Json>(std::move(number)) : std::nullopt;
    }

    std::optional<Json> read_list() {
        ++m_pos;
        Json list = Json::array();
        do {
            if (peek() == ']') {
                break;
            }
            auto value = read_value();
            if (!value) {
                return std::nullopt;
            }
            list.push_back(std::move(*value));
        } while (consume(','));
        if (!consume(']')) {
            return std::nullopt;
        }
        return list;
    }

    std::optional<Json> read_dict() {
        ++m_pos;
        Json dict = Json::object();
        do {
            const char c = peek();
            if (c == '}') {
                break;
            }
            if (c != '"' && c != '\'') {
                return std::nullopt;
            }
            auto key = read_string();
            if (!key || !consume(':')) {
                return std::nullopt;
            }
            auto value = read_value();
            if (!value) {
                return std::nullopt;
            }
            dict[*key] = std::move(*value);
        } while (consume(','));
        if (!consume('}')) {
            return std::nullopt;
        }
        return dict;
    }
};

// Returns the position closing the [ at pos, or npos if it is not closed. Counts all bracket types together:
// a list that is not a tool call can hold text that looks like one, e.g. in a string argument.
size_t find_closing_bracket(const std::string& text, size_t pos) {
    size_t depth = 0;
    char quote = '\0';
    for (size_t i = pos; i < text.size(); ++i) {
        const char c = text[i];
        if (quote != '\0') {
            if (c == '\\') {
                ++i;
            } else if (c == quote) {
                quote = '\0';
            }
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '[' || c == '(' || c == '{') {
            ++depth;
        } else if ((c == ']' || c == ')' || c == '}') && --depth == 0) {
            return i;
        }
    }
    return std::string::npos;
}

// foo[bar()] and [bar()](url) are an index and a link, not a list of calls
bool is_standalone_list(const std::string& text, size_t begin, size_t end) {
    if (begin > 0) {
        const char before = text[begin - 1];
        if (before == ')' || before == ']' || before == '_' || (before >= '0' && before <= '9') ||
            (before >= 'a' && before <= 'z') || (before >= 'A' && before <= 'Z')) {
            return false;
        }
    }
    const char after = end + 1 < text.size() ? text[end + 1] : '\0';
    return after != '(' && after != '[';
}

}  // namespace

class Llama3PythonicToolParser::Llama3PythonicToolParserImpl {
public:
    void parse(JsonContainer& message) {
        // Input example
        // string message = "[get_weather(location='New York, NY', unit='celsius')]<|eom_id|>";
        const std::string text = message["content"].get_string();

        // Text can hold other brackets before the calls, e.g. "See [1]: [f(a=1)]".
        // Stops at an unclosed bracket: what follows may be inside it, e.g. in output cut by max_new_tokens.
        for (size_t pos = text.find('['); pos != std::string::npos;) {
            const size_t end = find_closing_bracket(text, pos);
            if (end == std::string::npos) {
                return;
            }
            if (is_standalone_list(text, pos, end)) {
                if (auto calls = PythonicCallsReader(text, pos).read_calls()) {
                    const auto serialized = calls->dump(-1, ' ', false, Json::error_handler_t::replace);
                    message["tool_calls"] = JsonContainer::from_json_string(serialized);
                    return;
                }
            }
            pos = text.find('[', end + 1);
        }
    }
};

Llama3PythonicToolParser::Llama3PythonicToolParser() {
    m_impl = std::make_unique<Llama3PythonicToolParserImpl>();
}

void Llama3PythonicToolParser::parse(JsonContainer& message) {
    m_impl->parse(message);
}

Llama3PythonicToolParser::~Llama3PythonicToolParser() = default;

class Llama3JsonToolParser::Llama3JsonToolParserImpl {
public:
    void parse(JsonContainer& message) {
        // Find JSON in the message
        std::string msg_content = message["content"].get_string();

        size_t json_start = msg_content.find('{');
        size_t json_end = msg_content.rfind('}');
        if (json_start == std::string::npos || json_end == std::string::npos || json_end <= json_start) {
            return;
        }
        auto res = JsonContainer::array();
        res.push_back(JsonContainer::from_json_string(msg_content.substr(json_start, json_end - json_start + 1)));
        message["tool_calls"] = res;
    }
};

Llama3JsonToolParser::Llama3JsonToolParser() {
    m_impl = std::make_unique<Llama3JsonToolParserImpl>();
}

void Llama3JsonToolParser::parse(JsonContainer& message) {
    m_impl->parse(message);
}

Llama3JsonToolParser::~Llama3JsonToolParser() = default;

class ReasoningParser::ReasoningParserImpl {
public:
    ReasoningParserImpl(bool expect_open_tag,
                            bool keep_original_content,
                            const std::string& open_tag,
                            const std::string& close_tag):
    m_expect_open_tag(expect_open_tag),
    m_keep_original_content(keep_original_content),
    m_open_tag(open_tag),
    m_close_tag(close_tag) {};

    void parse(JsonContainer& message) {
        std::string reasoning_content;
        std::string content = message["content"].get_string();

        if (m_expect_open_tag) {
            const size_t open_pos = content.find(m_open_tag);
            const size_t close_pos = content.find(m_close_tag);

            if (open_pos != std::string::npos && close_pos != std::string::npos && close_pos > open_pos) {
                reasoning_content = content.substr(open_pos + m_open_tag.size(), close_pos - (open_pos + m_open_tag.size()));
                if (!m_keep_original_content) {
                     // Remove <think>...</think/> from content
                    message["content"] = content.substr(0, open_pos) + content.substr(close_pos + m_close_tag.size());
                }
            }
        } else {
            const size_t close_pos = content.find(m_close_tag);
            if (close_pos != std::string::npos) {
                reasoning_content = content.substr(0, close_pos);
                if (!m_keep_original_content) {
                     // Remove ...</think/> from content
                    message["content"] = content.substr(close_pos + m_close_tag.size());
                }
            } else {
                reasoning_content = content;
                if (!m_keep_original_content) {
                    message["content"] = "";
                }
            }
        }

        message["reasoning_content"] = reasoning_content;
    }
private:
    bool m_expect_open_tag;
    bool m_keep_original_content;
    std::string m_open_tag;
    std::string m_close_tag;
};

ReasoningParser::ReasoningParser(bool expect_open_tag, bool keep_original_content, const std::string& open_tag, const std::string& close_tag) {
    m_impl = std::make_unique<ReasoningParserImpl>(expect_open_tag, keep_original_content, open_tag, close_tag);
}

void ReasoningParser::parse(JsonContainer& message) {
    m_impl->parse(message);
}

ReasoningParser::~ReasoningParser() = default;

Parser::~Parser() = default;

} // namespace ov::genai
