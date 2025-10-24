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
    std::string m_text_cache;
    bool m_deactivated = false;
public:
    ReasoningParserImpl() = default;
    
    ReasoningParserImpl(bool expect_open_tag,
                    bool keep_original_content,
                    const std::string& open_tag, 
                    const std::string& close_tag)
        : m_expect_open_tag(expect_open_tag),
          m_keep_original_content(keep_original_content),
          m_open_tag(open_tag),
          m_close_tag(close_tag) {
        m_text_cache.reserve(close_tag.size());
    }

    std::string parse(
        JsonContainer&  message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens
    ) {
        if (m_deactivated) {
            return delta_text;
        }
        
        if (m_first_run) {
            m_first_run = false;
            if (!m_expect_open_tag) {
                m_think_tag_opened = true;
            }
        }

        // Initialize message fields if needed
        if (!message.contains("reasoning_content")) {
            message["reasoning_content"] = "";
        }
        if (!message.contains("content")) {
            message["content"] = "";
        }
        
        // Combine cached text with new delta
        m_text_cache += delta_text;
        const std::string& txt_chunk = m_text_cache;
        
        auto reason_str = message["reasoning_content"].get_string();

        if (!m_think_tag_opened && m_expect_open_tag) {
            // Look for opening tag
            size_t open_idx = txt_chunk.find(m_open_tag);
            if (open_idx != std::string::npos) {
                // Thinking has started
                m_think_tag_opened = true;
                size_t content_start = open_idx + m_open_tag.size();
                
                // Check if closing tag is also present
                size_t close_idx = txt_chunk.find(m_close_tag, content_start);
                if (close_idx != std::string::npos) {
                    // Both tags in same chunk
                    reason_str = txt_chunk.substr(content_start, close_idx - content_start);
                    message["reasoning_content"] = reason_str;
                    
                    if (!m_keep_original_content) {
                        delta_text = txt_chunk.substr(close_idx + m_close_tag.size());
                    }
                    
                    m_think_tag_opened = false;
                    m_deactivated = true;
                    m_text_cache.clear();
                } else {
                    // Only opening tag found
                    reason_str += txt_chunk.substr(content_start);
                    message["reasoning_content"] = reason_str;
                    
                    if (!m_keep_original_content) {
                        delta_text.clear();
                    }
                    m_text_cache.clear();
                }
                return delta_text;
            }
            // Opening tag not found, keep accumulating
            return delta_text;
        } 
        
        if (m_think_tag_opened) {
            // Look for closing tag
            size_t close_idx = txt_chunk.find(m_close_tag);
            if (close_idx != std::string::npos) {
                // Thinking tag was closed
                reason_str += txt_chunk.substr(0, close_idx);
                message["reasoning_content"] = reason_str;
                
                if (!m_keep_original_content) {
                    delta_text = txt_chunk.substr(close_idx + m_close_tag.size());
                }
                
                m_text_cache.clear();
                m_think_tag_opened = false;
                m_deactivated = true;
                return delta_text;
            }
            
            // Closing tag not found - check if end might be partial match
            size_t num_chars_to_keep = find_prefix_match_length(txt_chunk, m_close_tag);
            
            if (num_chars_to_keep > 0) {
                // Keep potential partial match in cache
                reason_str += txt_chunk.substr(0, txt_chunk.size() - num_chars_to_keep);
                message["reasoning_content"] = reason_str;
                m_text_cache = txt_chunk.substr(txt_chunk.size() - num_chars_to_keep);
            } else {
                // No partial match - add all to reasoning
                reason_str += txt_chunk;
                message["reasoning_content"] = reason_str;
                m_text_cache.clear();
            }
            
            if (!m_keep_original_content) {
                delta_text.clear();
            }
        }
        // else: accumulating text before opening tag
        
        return delta_text;
    }

    // Find the longest suffix of txt that is a prefix of close_tag
    size_t find_prefix_match_length(const std::string& txt, const std::string& close_tag) const {
        size_t max_check = std::min(txt.size(), close_tag.size() - 1);
        for (size_t len = max_check; len >= 1; --len) {
            if (txt.compare(txt.size() - len, len, close_tag, 0, len) == 0) {
                return len;
            }
        }
        return 0;
    }

public:
    void reset() {
        m_first_run = true;
        m_think_tag_opened = false;
        m_text_cache.clear();
        m_deactivated = false;
    }
};

ReasoningIncrementalParser::ReasoningIncrementalParser(bool expect_open_tag, bool keep_original_content, const std::string& open_tag, const std::string& close_tag) {
    m_impl = std::make_unique<ReasoningParserImpl>(expect_open_tag, keep_original_content, open_tag, close_tag);
}

ReasoningIncrementalParser::~ReasoningIncrementalParser() = default;

std::string ReasoningIncrementalParser::parse(
    JsonContainer& message,
    std::string& delta_text,
    const std::optional<std::vector<int64_t>>& delta_tokens
) {
    return m_impl->parse(message, delta_text, delta_tokens);
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
