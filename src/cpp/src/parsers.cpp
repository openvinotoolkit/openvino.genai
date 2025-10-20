#include "openvino/genai/parsers.hpp"
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cctype>
#include <stdexcept>

namespace ov::genai {

class ReasoningParser::ReasoningParserImpl {
private:
    bool m_expect_open_tag;
    bool m_first_run = true;
    bool m_keep_original_content;
    bool m_think_tag_opened = false;
    std::string m_open_tag;
    std::string m_close_tag;
    std::string m_text_cache = "";
    std::map<std::string, std::string> accumulated_parsed;
public:
    bool m_deactivated = false;
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
        JsonContainer&  msg,
        const std::string& previous_text, 
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& previous_tokens, 
        const std::optional<std::vector<int64_t>>& delta_tokens
    ) {
        if (m_deactivated) {
            return delta_text;
        }
        if (m_expect_open_tag && m_first_run) {
            m_think_tag_opened = true;
        }
        m_first_run = false;

        if (!msg.contains("reasoning_content")) {
            msg["reasoning_content"] = "";
        }
        if (!msg.contains("content")) {
            msg["content"] = "";
        }
        

        auto txt_chunk = m_text_cache + delta_text;
        auto reason_str = msg["reasoning_content"].get_string();
        auto content_str = msg["content"].get_string();

        if (!m_think_tag_opened && txt_chunk.find(m_open_tag) != std::string::npos && !m_expect_open_tag) {
            OPENVINO_ASSERT(m_open_tag.find(m_text_cache) != std::string::npos, "m_text_cache should be a prefix of m_open_tag");
            
            // Thinking has started
            auto open_idx = txt_chunk.find(m_open_tag);
            reason_str += txt_chunk.substr(open_idx + std::string(m_open_tag).size(), txt_chunk.size() - (open_idx + std::string(m_open_tag).size()));
            if (!m_keep_original_content) {
                delta_text = "";
            }
            
            m_think_tag_opened = true;
            msg["reasoning_content"] = reason_str;
            m_text_cache = "";

            if (txt_chunk.find(m_close_tag) != std::string::npos) {
                // If <think> and </think> are in the same txt_chunk + delta_text
                auto close_idx = txt_chunk.find(m_close_tag);
                reason_str = txt_chunk.substr(open_idx + std::string(m_open_tag).size(), close_idx - (open_idx + std::string(m_open_tag).size()));
                content_str = txt_chunk.substr(close_idx + std::string(m_close_tag).size(), txt_chunk.size() - (close_idx + std::string(m_close_tag).size()));
                if (!m_keep_original_content) {
                    delta_text = content_str;
                }
                m_think_tag_opened = false;
                m_deactivated = true;
                msg["reasoning_content"] = reason_str;
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
                delta_text = txt_chunk.substr(close_idx + std::string(m_close_tag).size(), txt_chunk.size() - (close_idx + std::string(m_close_tag).size()));
            }

            msg["reasoning_content"] = reason_str;
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
            msg["reasoning_content"] = reason_str;
        } else {
            // Think tag was not opened yet and not found in the current delta_text.
            // Accumulate text in the cache to detect if <think> is split between several delta_text pieces.
            m_text_cache += delta_text;
        }
        
        return delta_text;
    }
};

ReasoningParser::ReasoningParser(bool expect_open_tag, bool keep_original_content, const std::string& open_tag, const std::string& close_tag) {
    m_impl = std::make_shared<ReasoningParserImpl>(expect_open_tag, keep_original_content, open_tag, close_tag);
}

std::string ReasoningParser::parse(
    JsonContainer& msg,
    const std::string& previous_text, 
    std::string& delta_text,
    const std::optional<std::vector<int64_t>>& previous_tokens, 
    const std::optional<std::vector<int64_t>>& delta_tokens
) {
    return m_impl->parse(msg, previous_text, delta_text, previous_tokens, delta_tokens);
}

class Llama32PythonicToolParser::Llama32PythonicToolParserImpl {
public:
    Llama32PythonicToolParserImpl(bool keep_original_content) : m_keep_original_content(keep_original_content) {}
    bool m_keep_original_content;

    void parse(JsonContainer& input) {
        // Input example
        // string input = "[get_weather(location='New York, NY', unit='celsius')]<|eom_id|>";

        // Regex to capture the [...] part
        std::smatch m;
        const std::string& text = input["content"].get_string();
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
        // Parse arguments of the form key='value'
        std::regex arg_re(R"((\w+)\s*=\s*\"([^"]*)\")");
        auto it = std::sregex_iterator(args.begin(), args.end(), arg_re);
        for (; it != std::sregex_iterator(); ++it) {
            kv[std::string((*it)[1])] = std::string((*it)[2]);
        }
        
        // Split function name and arguments
        input["tool_calls"] = JsonContainer::array();
        input["tool_calls"].push_back(JsonContainer({{"name", name}, {"arguments", kv}}));
        
        if (!m_keep_original_content) {
            input["content"] = regex_replace(text, r, "");
        }
    }
};

Llama32PythonicToolParser::Llama32PythonicToolParser(bool keep_original_content) {
    m_impl = std::make_shared<Llama32PythonicToolParserImpl>(keep_original_content);
}

void Llama32PythonicToolParser::parse(JsonContainer& input) {
    m_impl->parse(input);
}

class Llama32JsonToolParser::Llama32JsonToolParserImpl {
private:
    bool m_keep_original_content;
public:
    Llama32JsonToolParserImpl(bool keep_original_content) : m_keep_original_content(keep_original_content) {}

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
        
        if (!m_keep_original_content) {
            message["content"] = msg_content.substr(0, json_start) + msg_content.substr(json_end + 1);
        }
    }
};

Llama32JsonToolParser::Llama32JsonToolParser(bool keep_original_content) {
    m_impl = std::make_shared<Llama32JsonToolParserImpl>(keep_original_content);
}

void Llama32JsonToolParser::parse(JsonContainer& input) {
    m_impl->parse(input);
}

class BaseReasoningParser::BaseReasoningParserImpl {
public:
    BaseReasoningParserImpl(bool expect_open_tag,
                            bool keep_original_content,
                            const std::string& open_tag,
                            const std::string& close_tag):
    m_expect_open_tag(expect_open_tag),
    m_keep_original_content(keep_original_content),
    m_open_tag(open_tag),
    m_close_tag(close_tag) {};

    void parse(JsonContainer& input) {
        std::string reasoning_content;
        std::string content = input["content"].get_string();

        size_t start = content.find(m_open_tag);
        size_t end = content.find(m_close_tag);

        if (start != std::string::npos && end != std::string::npos && end > start) {
            reasoning_content = content.substr(start + m_open_tag.size(), end - (start + m_open_tag.size()));
            if (!m_keep_original_content) {
                // Remove <think>...</think/> from content
                input["content"] = content.substr(0, start) + content.substr(end + m_close_tag.size());
            }
        } else {
            reasoning_content = "";
        }

        input["reasoning_content"] = reasoning_content;
    }
private:
    bool m_expect_open_tag;
    bool m_keep_original_content;
    std::string m_open_tag;
    std::string m_close_tag;
};

BaseReasoningParser::BaseReasoningParser(bool expect_open_tag, bool keep_original_content, const std::string& open_tag, const std::string& close_tag) {
    m_impl = std::make_shared<BaseReasoningParserImpl>(expect_open_tag, keep_original_content, open_tag, close_tag);
}

void BaseReasoningParser::parse(JsonContainer& input) {
    m_impl->parse(input);
}

} // namespace ov::genai
