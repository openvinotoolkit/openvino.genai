
#include "openvino/genai/parsers/lfm2_tool_parser.hpp"

namespace ov::genai {


class Lfm2Parser::Lfm2ParserImpl {
public:
    void parse(JsonContainer& message, const std::optional<std::vector<int64_t>>& /*tokens*/) {
        std::string msg_content = message["content"].get_string();
        auto res = JsonContainer::array();
        res.push_back(JsonContainer::from_json_string("{\"my tool\":\"test\"}"));
        message["tool_calls"] = res;
    }
};


Lfm2Parser::Lfm2Parser() {
    m_impl = std::make_unique<Lfm2ParserImpl>();
}

void Lfm2Parser::parse(JsonContainer& message, const std::optional<std::vector<int64_t>>& tokens) {
    m_impl->parse(message, tokens);
}

std::string Lfm2Parser::parseChunk(JsonContainer& delta_message,
                                   std::string& delta_text,
                                   const std::optional<std::vector<int64_t>>& /*delta_tokens*/) {
    return delta_text;
}

void Lfm2Parser::reset() {
    // Stateless parser; nothing to reset.
}

Lfm2Parser::~Lfm2Parser() = default;

}