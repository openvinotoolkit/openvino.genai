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
        JsonContainer&  message,
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

        if (!message.contains("reasoning_content")) {
            message["reasoning_content"] = "";
        }
        if (!message.contains("content")) {
            message["content"] = "";
        }
        

        auto txt_chunk = m_text_cache + delta_text;
        auto reason_str = message["reasoning_content"].get_string();
        auto content_str = message["content"].get_string();

        if (!m_think_tag_opened && txt_chunk.find(m_open_tag) != std::string::npos && m_expect_open_tag) {
            // Thinking has started
            auto open_idx = txt_chunk.find(m_open_tag);
            
            reason_str += txt_chunk.substr(open_idx + m_open_tag.size(), txt_chunk.size() - (open_idx + m_open_tag.size()));
            if (!m_keep_original_content) {
                delta_text = "";
            }
            
            m_think_tag_opened = true;
            message["reasoning_content"] = reason_str;
            m_text_cache = "";

            if (txt_chunk.find(m_close_tag) != std::string::npos) {
                // If <think> and </think> are in the same txt_chunk + delta_text
                auto close_idx = txt_chunk.find(m_close_tag);
                reason_str = txt_chunk.substr(open_idx + m_open_tag.size(), close_idx - (open_idx + m_open_tag.size()));
                content_str = txt_chunk.substr(close_idx + m_close_tag.size(), txt_chunk.size() - (close_idx + m_close_tag.size()));
                if (!m_keep_original_content) {
                    delta_text = content_str;
                }
                m_think_tag_opened = false;
                m_deactivated = true;
                message["reasoning_content"] = reason_str;
            }
        } else if (m_think_tag_opened && txt_chunk.find(m_close_tag) != std::string::npos) {
            // Thinking tag was closed
            auto close_idx = txt_chunk.find(m_close_tag);

            reason_str += txt_chunk.substr(0, close_idx);
            if (!m_keep_original_content) {
                // Cut from the txt_chunk which is before </think> and leave only what is after </think>.
                // Example if m_text_cache + delta_text = "...some text</th" + "ink>Answer is 3" = "...some text</think>Answer is 3"
                // we want to keep in delta_txt only "Answer is 3". 
                // We can operate with txt_chunk since final characters closing the tag ("ink>") are always in delta_text.
                delta_text = txt_chunk.substr(close_idx + m_close_tag.size(), txt_chunk.size() - (close_idx + m_close_tag.size()));
            }

            message["reasoning_content"] = reason_str;
            m_text_cache = "";
            m_think_tag_opened = false;
            m_deactivated = true;
        } else if (m_think_tag_opened) {
            // Thinking tag was already opened and not closed yet
            
            // If we have subsequently "sdf</th", "i", "nk> The"
            // Then we put "sdf" to reason_str and "</th" to m_text_cache since it's a substring of close tag "</think>"
            // then we put "i" to m_text_cache since m_text_cache + delta_text = "</thi" is a substring of "</think>"
            // then (in the closing tag IF-block) we leave only " The" in delta_text.

            // If we have "ing. <", " 20 ", "40>"
            // Then we put "ing. " to reason_str and "<" to m_text_cache since it's a substring of close tag "</think>"
            // but since continuation " 20 " is not a substring of "</think>", we will end up in this IF-block again
            // and put " 20 " to reason_str and clear m_text_cache.

            // number of characters from the end of txt_chunk which can be part of the closing tag
            size_t num_chars_to_keep = 0; 
            // We must be sure that no chunks with the closing tag are included to reason_str.
            for (size_t i = txt_chunk.size(); i >= 1; --i) {
                // Get the substring of the i last characters of txt_chunk
                auto suffix = txt_chunk.substr(txt_chunk.size() - i, i);
                // If this suffix is a prefix of m_close_tag, we need to keep it in the cache.
                if (m_close_tag.find(suffix) == 0) {
                    num_chars_to_keep = i;
                    break;
                }
            }

            // If the suffix is a prefix of m_close_tag, we store it in the cache to detect if </think> is split between several delta_text pieces.
            if (num_chars_to_keep > 0) {
                m_text_cache = txt_chunk.substr(txt_chunk.size() - num_chars_to_keep, num_chars_to_keep);
                reason_str += txt_chunk.substr(0, txt_chunk.size() - num_chars_to_keep);
            } else {
                reason_str += txt_chunk;
                m_text_cache = "";
            }

            if (!m_keep_original_content) {
                delta_text = "";
            }
            message["reasoning_content"] = reason_str;
        } else {
            // Think tag was not opened yet and not found in the current delta_text.
            // Accumulate text in the cache to detect if <think> is split between several delta_text pieces.
            m_text_cache += delta_text;
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
    void parse(JsonContainer& message) {
        // Input example
        // string message = "[get_weather(location='New York, NY', unit='celsius')]<|eom_id|>";

        // Regex to capture the [...] part
        std::smatch m;
        const std::string& text = message["content"].get_string();
        std::regex r(R"(\[.*?\])");
        if (!std::regex_search(text, m, r)) {
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
