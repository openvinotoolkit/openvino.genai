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

std::string state_to_string(const ParsingState state) {
    switch (state) {
        case ParsingState::CONTENT:
            return "CONTENT";
        case ParsingState::REASONING:
            return "REASONING";
        case ParsingState::TOOL_CALLING:
            return "TOOL_CALLING";
        case ParsingState::UNDEFINED:
            return "UNDEFINED";
        default:
            return "UNKNOWN";
    }
}

class DeepSeekR1Parser : public IncrementalParserBase {
private:
    bool m_starts_with_thinking = true;
    ParsingState m_parsing_state = ParsingState::REASONING;
public:
    DeepSeekR1Parser() = default;
    std::map<std::string, std::string> accumulated_parsed;

    ParsedMessage parse(
        const std::string& previous_text, 
        const std::string& delta_text,
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt) {
        ParsedMessage msg;

        if (!m_starts_with_thinking) {
            m_parsing_state = ParsingState::UNDEFINED;
        } else {
            m_parsing_state = ParsingState::REASONING;
        }

        if (m_parsing_state == ParsingState::UNDEFINED && delta_text.find("<think>") != std::string::npos) {
            m_parsing_state = ParsingState::REASONING;
            auto think_idx = delta_text.find("<think>");
            msg["reasoning_content"] = delta_text.substr(think_idx + std::string("<think>").size(), delta_text.size() - (think_idx + std::string("<think>").size()));
        } else if (delta_text.find("</think>") != std::string::npos && m_parsing_state == ParsingState::REASONING) {
            auto think_idx = delta_text.find("</think>");

            msg["reasoning_content"] = delta_text.substr(0, think_idx);
            msg["content"] = delta_text.substr(think_idx + std::string("</think>").size(), delta_text.size() - (think_idx + std::string("</think>").size()));

            m_parsing_state = ParsingState::CONTENT;
        } else if (m_parsing_state == ParsingState::REASONING) {
            msg["reasoning_content"] = delta_text;
        } else if (m_parsing_state == ParsingState::CONTENT) {
            msg["content"] = delta_text;
        } else {
            throw std::runtime_error("Unexpected state in DeepSeekR1Parser");
        }
        msg["state"] = state_to_string(m_parsing_state);
        
        // TODO: consider accumulating all fiels and returning accumulated fields instead of parsing once more at the end.
        
        // std::string accumulated_reasoning += msg["reasoning_content"];
        accumulated_parsed["content"] += msg["content"];
        
        // accumulated_parsed["reasoning_content"] = accumulated_reasoning;
        // TODO: if thinking is closed, disable parsing and give content without cutting thinking.
        return msg;
    }
};


ParsedMessage Llama32PythonicParser::parse(ParsedMessage& input) {
    // Input example
    // string input = "[get_weather(location='New York, NY', unit='celsius')]<|eom_id|>";

    // Regex to capture the [...] part
    smatch m;
    const std::string& text = input.at("content");
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
        input["tool_calls"] = j.dump();
        return input;
    }
    return ParsedMessage{};
}

ParsedMessage BaseReasoningParser::parse(ParsedMessage& input) {
    ParsedMessage res;
    std::string reasoning_content;
    const std::string& content = input.at("content");
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


TextParserStreamer::TextParserStreamer(const Tokenizer& tokenizer) 
    : ov::genai::TextStreamer(tokenizer, [this](std::string s) -> ov::genai::CallbackTypeVariant {
                return this->write(s);
    }) {
        m_reasoning_parser = std::make_shared<DeepSeekR1Parser>();
    }

StreamingStatus TextParserStreamer::write(ParsedMessage& message) {
    return StreamingStatus::RUNNING;
}

ov::genai::CallbackTypeVariant TextParserStreamer::write(std::string message) {
    // for (auto& parser: m_parsers) {
    //     if (parser.is_active()) {
    //         msg = parser.parse(m_text_buffer, message, msg);
    //     }
    // }

    // m_text_buffer += message;
    // return write(msg);
}

} // namespace ov::genai
