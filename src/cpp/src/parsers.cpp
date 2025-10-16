#include "openvino/genai/parsers.hpp"
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cctype>
#include <stdexcept>
#include <bits/stdc++.h>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

namespace ov::genai {

class ReasoningParserImpl {
private:
    bool m_starts_with_thinking = true;
    bool m_keep_original_content = true;
    bool m_think_tag_opened = false;
    std::string m_open_tag = "<think>";
    std::string m_close_tag = "</think>";
    std::string m_text_cache = "";
    std::map<std::string, std::string> accumulated_parsed;
public:
    bool m_deactivated = false;
    ReasoningParserImpl() = default;
    ReasoningParserImpl(bool starts_with_thinking = true,
                    bool keep_original_content = true)
        : m_starts_with_thinking(starts_with_thinking),
          m_keep_original_content(keep_original_content) {}

    std::string parse(
        ParsedMessage&  msg,
        const std::string& previous_text, 
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& previous_tokens, 
        const std::optional<std::vector<int64_t>>& delta_tokens
    ) {
        if (!msg.contains("reasoning_content")) {
            msg["reasoning_content"] = "";
        }
        if (!msg.contains("content")) {
            msg["content"] = "";
        }
        
        bool think_tag_closed = delta_text.find(m_close_tag) != std::string::npos;
        if (m_starts_with_thinking) {
            m_think_tag_opened = true;
        }

        auto txt_chunk = m_text_cache + delta_text;
        auto reason_str = msg["reasoning_content"].get_string();
        auto content_str = msg["content"].get_string();

        if (!m_think_tag_opened && txt_chunk.find(m_open_tag) != std::string::npos && !m_starts_with_thinking) {
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
            // content_str += txt_chunk.substr(close_idx + std::string(m_close_tag).size(), txt_chunk.size() - (close_idx + std::string(m_close_tag).size()));
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
            // reason_str += m_text_cache
            
            // "sdf</th", "i", "nk>"
            // "sd<", " 2"
            
            // m_text_cache = "<"
            // delta_text = " 2"

            size_t num_chars_to_keep = 0; // number of characters from the end of txt_chunk which can be part of the closing tag
            // We must be sure that no chunks with the closing tag are included to reason_str.
            for (size_t i = txt_chunk.size(); i >= 1; --i) {
                // Get the substring of the i last characters of txt_chunk
                auto suffix = txt_chunk.substr(txt_chunk.size() - i, i);
                if (m_close_tag.find(suffix) != std::string::npos) {
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

ReasoningParser::ReasoningParser(bool starts_with_thinking, bool keep_original_content) {
    m_impl = std::make_shared<ReasoningParserImpl>(starts_with_thinking, keep_original_content);
}

std::string ReasoningParser::parse(
    ParsedMessage& msg,
    const std::string& previous_text, 
    std::string& delta_text,
    const std::optional<std::vector<int64_t>>& previous_tokens, 
    const std::optional<std::vector<int64_t>>& delta_tokens
) {
    return m_impl->parse(msg, previous_text, delta_text, previous_tokens, delta_tokens);
}

bool ReasoningParser::is_active() const {
    return !m_impl->m_deactivated;
}

ParsedMessage Llama32PythonicToolParser::parse(ParsedMessage& input) {
    // Input example
    // string input = "[get_weather(location='New York, NY', unit='celsius')]<|eom_id|>";

    // Regex to capture the [...] part
    smatch m;
    const std::string& text = input["content"].get_string();
    regex r(R"(\[.*?\])");
    if (regex_search(text, m, r)) {
        // Strip outer [ ]
        string call = m.str().substr(1, m.str().size() - 2);

        // Split function name and arguments
        size_t pos = call.find('(');
        string name = call.substr(0, pos);
        string args = call.substr(pos + 1, call.size() - pos - 2); // inside (...)

        // Parse arguments of the form key='value'
        map<string, string> kv;
        regex arg_re(R"((\w+)\s*=\s*'([^']*)')");
        auto it = sregex_iterator(args.begin(), args.end(), arg_re);
        for (; it != sregex_iterator(); ++it) {
            kv[(*it)[1]] = (*it)[2];
        }
        json j = json::array({{
            {"name", name},
            {"arguments", kv}
        }});
        if (!m_keep_original_content) {
            input["content"] = regex_replace(text, r, "");
        }
        std::cout << j.dump() << std::endl;
        input["tool_calls"] = ParsedMessage::from_json_string(j.dump());
        return input;
    }
    return ParsedMessage{};
}

ParsedMessage BaseReasoningParser::parse(ParsedMessage& input) {
    ParsedMessage res;
    std::string reasoning_content;
    // auto content = input["content"];
    std::string content = input["content"].get_string();
    res["content"] = content;

    size_t start = content.find(m_open_tag);
    size_t end = content.find(m_close_tag);

    if (start != std::string::npos && end != std::string::npos && end > start) {
        reasoning_content = content.substr(start + m_open_tag.size(), end - (start + m_open_tag.size()));
        if (!m_keep_original_content) {
            // Remove <think>...</think/> from content
            res["content"] = content.substr(0, start) + content.substr(end + m_close_tag.size());
        }
    } else {
        reasoning_content = "";
    }

    res["reasoning_content"] = reasoning_content;
    return res;
}

std::map<std::string, std::function<std::shared_ptr<IncrementalParserBase>()>> registered_incremental_parsers;
std::map<std::string, std::function<std::shared_ptr<ParserBase>()>> registered_base_parsers;

// static initializer to register available buildin parsers
static bool register_backends() {
    registered_incremental_parsers[DeepSeekR1ReasoningParser::name()] = []() { return std::make_shared<DeepSeekR1ReasoningParser>(/*starts_with_thinking*/ true); };
    registered_incremental_parsers[Phi4ReasoningParser::name()] = []() { return std::make_shared<Phi4ReasoningParser>(/*starts_with_thinking*/ false); };
    
    registered_base_parsers[Llama32PythonicToolParser::name()] = []() { return std::make_shared<Llama32PythonicToolParser>(); };

    // TODO: Add more parsers and register them here.
    return true;
}

// Ensure the backends are registered before main
static bool are_backends_registered = register_backends();

std::shared_ptr<IncrementalParserBase> IncrementalParserBase::get_parser(std::string name) {
    if (!are_backends_registered) {
        register_backends();
    }

    if (registered_incremental_parsers.find(name) != registered_incremental_parsers.end()) {
        return registered_incremental_parsers[name]();
    }
    return nullptr;
}

std::shared_ptr<ParserBase> ParserBase::get_parser(std::string name) {
    if (!are_backends_registered) {
        register_backends();
    }

    if (registered_base_parsers.find(name) != registered_base_parsers.end()) {
        return registered_base_parsers[name]();
    }
    return nullptr;
}

static std::vector<std::string> get_parsers_names() {
    std::vector<std::string> names;
    for (const auto& [name, _] : registered_incremental_parsers) {
        names.push_back(name);
    }
    for (const auto& [name, _] : registered_base_parsers) {
        names.push_back(name);
    }
    return names;
}



} // namespace ov::genai
