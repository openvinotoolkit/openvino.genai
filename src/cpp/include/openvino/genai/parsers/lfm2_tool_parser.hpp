// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../parsers.hpp"

namespace ov {
namespace genai {



/**
 * @brief Parser for Lfm2 calls format.
 *
 * Lfm2Parser extracts tool calls from text content formatted
 * in Lfm2's style, e.g. {"type": "function", "function": {"name": "get_weather", "parameters": {"location": "New York, NY", ...}}}.
 * It does not modify the original content, only extracts and adds tool call information to the message.
 */
class OPENVINO_GENAI_EXPORTS Lfm2Parser : public Parser {
public:
    Lfm2Parser();
    ~Lfm2Parser();
    
    /**
     * @brief Parse Lfm2 tool calls from text.
     *
     * Extracts tool call information from text formatted in Lfm2's style
     * and adds the tool calls to the JsonContainer without modifying the original content.
     *
     * @param message JsonContainer containing the text to parse and to store tool call results
     * @param tokens Optional vector of token IDs associated with the text, which can be used for more efficient token-based parsing if supported by the implementation.
     */ 
    void parse(JsonContainer& message, const std::optional<std::vector<int64_t>>& tokens = std::nullopt) override;

    std::string parseChunk(
        JsonContainer& delta_message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override;

    void reset() override;
private:
    class Lfm2ParserImpl;
    std::unique_ptr<Lfm2ParserImpl> m_impl;
};


}  // namespace genai
}  // namespace ov