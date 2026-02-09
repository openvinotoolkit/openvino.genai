#include "openvino/genai/parsers.hpp"
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cctype>
#include <stdexcept>

namespace ov::genai {

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

class Llama3PythonicToolParser::Llama3PythonicToolParserImpl {
public:
    std::regex m_pattern = std::regex(R"(\[(.*?)\])");
    void parse(JsonContainer& message) {
        // Input example
        // string message = "[get_weather(location='New York, NY', unit='celsius')]<|eom_id|>";

        // Regex to capture the [...] part
        std::smatch m;
        const std::string& text = message["content"].get_string();

        if (!std::regex_search(text, m, m_pattern)) {
            return;
        }

        // Strip outer [ ]
        std::string call = m.str().substr(1, m.str().size() - 2);

        size_t pos = call.find('(');
        std::string name = call.substr(0, pos);
        std::string args = call.substr(pos + 1, call.size() - pos - 2); // inside (...)
        
        JsonContainer kv;
        // Parse arguments of the form key="value"
        std::regex arg_re(R"((\w+)\s*=\s*\"([^"]*)\")");
        auto it = std::sregex_iterator(args.begin(), args.end(), arg_re);
        for (; it != std::sregex_iterator(); ++it) {
            kv[std::string((*it)[1])] = std::string((*it)[2]);
        }
        
        // Split function name and arguments
        message["tool_calls"] = JsonContainer::array();
        message["tool_calls"].push_back(JsonContainer({{"name", name}, {"arguments", kv}}));
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

        size_t start = content.find(m_open_tag);
        size_t end = content.find(m_close_tag);

        if (start != std::string::npos && end != std::string::npos && end > start) {
            reasoning_content = content.substr(start + m_open_tag.size(), end - (start + m_open_tag.size()));
            if (!m_keep_original_content) {
                // Remove <think>...</think/> from content
                message["content"] = content.substr(0, start) + content.substr(end + m_close_tag.size());
            }
        } else {
            reasoning_content = "";
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
