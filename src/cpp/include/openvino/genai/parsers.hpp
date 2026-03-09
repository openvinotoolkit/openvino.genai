// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <memory>
#include <vector>
#include "openvino/genai/json_container.hpp"

namespace ov {
namespace genai {

/**
 * @brief Abstract base class for parsers that process complete text content at the end of generation.
 */
class OPENVINO_GENAI_EXPORTS Parser {
public:
    Parser() = default;
    virtual ~Parser();
    
    /**
     * @brief Parse complete text content at the end of generate call.
     *
     * This method processes the entire text content and extracts or modifies
     * information as needed. The results are stored in the provided JsonContainer.
     *
     * @param message JsonContainer containing the text to parse and to store results
     * @param tokens Optional vector of token IDs associated with the text, which can be used for more efficient token-based parsing if supported by the implementation.
     */
    virtual void parse(JsonContainer& message, const std::optional<std::vector<int64_t>>& tokens = std::nullopt) = 0;

        /**
     * @brief Parse incremental text content and return filtered text.
     *
     * This method processes incoming delta_text and returns filtered text that should
     * be added to the content.
     *
     * @param delta_message JsonContainer to store parsed results and metadata
     * @param delta_text New text chunk to be processed in this step and modified in place
     * @param delta_tokens Optional vector of new token IDs to be processed in case if more fast token-based processing is needed.
     * @return std::string filtered text that should be added to the 'content'
     */
    virtual std::string parseChunk(
        JsonContainer& delta_message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) = 0;

    /**
     * @brief Reset the internal state of the parser.
     */
    virtual void reset() = 0;
};

class OPENVINO_GENAI_EXPORTS ReasoningParser : public Parser {
public:
    /**
     * @brief ReasoningParser extracts reasoning content between open and close tags from text.
     * Field 'content' should be filled in order to extract reasoning content.
     * The reasoning content is stored in the 'reasoning_content' field of the JsonContainer.
     *
     * @param expect_open_tag If true then open_tag is expected to be generated, if false then it's already part of the model input string
     * @param keep_original_content Whether to preserve the original 'content' including reasoning sections
     * @param open_tag The opening tag (default: "<think>")
     * @param close_tag The closing tag (default: "</think>")
     */
    ReasoningParser(
        bool expect_open_tag = true,
        bool keep_original_content = true,
        const std::string& open_tag = "<think>",
        const std::string& close_tag = "</think>");

    ReasoningParser(ReasoningParser&&) noexcept;
    ReasoningParser& operator=(ReasoningParser&&) noexcept;
    
    /**
     * @brief Parse complete text content at the end of generate call.
     *
     * This method processes the entire text content and extracts or modifies
     * information as needed. The results are stored in the provided JsonContainer.
     *
     * @param message JsonContainer containing the text to parse and to store results
     * @param tokens Optional vector of token IDs associated with the text, which can be used for more efficient token-based parsing if supported by the implementation.
     */
    void parse(JsonContainer& message, const std::optional<std::vector<int64_t>>& tokens = std::nullopt) override;

    /**
     * @brief Parse incremental text content and return filtered text.
     *
     * ReasoningParser does not implement streaming-specific filtering.
     * This default implementation returns the incoming delta_text unchanged.
     */
    std::string parseChunk(
        JsonContainer& delta_message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override;

    /**
     * @brief Reset the internal state of the parser.
     *
     * ReasoningParser does not maintain streaming state in this default implementation.
     */
    void reset() override;

    ~ReasoningParser();
private:
    class ReasoningParserImpl;
    std::unique_ptr<ReasoningParserImpl> m_impl;
};

/**
 * @brief Parser for DeepSeek R1 model reasoning format.
 *
 * DeepSeekR1ReasoningParser is configured for the DeepSeek R1 model's reasoning format, which doesn't expect an opening tag.
 */
class OPENVINO_GENAI_EXPORTS DeepSeekR1ReasoningParser : public ReasoningParser {
public:
    DeepSeekR1ReasoningParser() : ReasoningParser(/*expect_open_tag=*/false) {};
};

/**
 * @brief Parser for Phi-4 model reasoning format.
 *
 * Phi4ReasoningParser is configured specifically for the Phi-4 model's reasoning format, which expects an opening tag by default.
 */
class OPENVINO_GENAI_EXPORTS Phi4ReasoningParser : public ReasoningParser {
public:
   Phi4ReasoningParser() : ReasoningParser(/*expect_open_tag=*/true) {};
};

/**
 * @brief Parser for Llama 3 Pythonic tool calls format.
 *
 * Llama3PythonicToolParser extracts tool calls from text content formatted
 * in Llama 3's Pythonic style, e.g. [get_weather(location='New York, NY', unit='celsius')].
 * It does not modify the original content,
 * only extracts and adds tool call information to the message.
 */
class OPENVINO_GENAI_EXPORTS Llama3PythonicToolParser : public Parser {
public:
    Llama3PythonicToolParser();
    ~Llama3PythonicToolParser();
    
    /**
     * @brief Parse Llama 3 Pythonic tool calls from text.
     *
     * Extracts tool call information from text formatted in Llama 3's Pythonic style
     * and adds the 'tool_calls' to the JsonContainer without modifying the original content.
     *
     * @param message JsonContainer containing the text to parse and to store tool call results
     * @param tokens Optional vector of token IDs associated with the text, which can be used for more efficient token-based parsing if supported by the implementation.
     */
    void parse(JsonContainer& message, const std::optional<std::vector<int64_t>>& tokens = std::nullopt) override;

    /**
     * @brief Parse incremental text content for Llama 3 Pythonic tool calls.
     *
     * Current implementation does not perform incremental tool-call extraction and
     * returns the input delta_text unchanged.
     */
    std::string parseChunk(
        JsonContainer& delta_message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override {
        (void)delta_message;
        (void)delta_tokens;
        return delta_text;
    }

    /**
     * @brief Reset internal parser state.
     *
     * Current implementation has no internal state to reset.
     */
    void reset() override {}
private:
    class Llama3PythonicToolParserImpl;
    std::unique_ptr<Llama3PythonicToolParserImpl> m_impl;
};

/**
 * @brief Parser for Llama 3 JSON tool calls format.
 *
 * Llama3JsonToolParser extracts tool calls from text content formatted
 * in Llama 3's JSON style, e.g. {"type": "function", "function": {"name": "get_weather", "parameters": {"location": "New York, NY", ...}}}.
 * It does not modify the original content, only extracts and adds tool call information to the message.
 */
class OPENVINO_GENAI_EXPORTS Llama3JsonToolParser : public Parser {
public:
    Llama3JsonToolParser();
    ~Llama3JsonToolParser();
    
    /**
     * @brief Parse Llama 3 JSON tool calls from text.
     *
     * Extracts tool call information from text formatted in Llama 3's JSON style
     * and adds the tool calls to the JsonContainer without modifying the original content.
     *
     * @param message JsonContainer containing the text to parse and to store tool call results
     * @param tokens Optional vector of token IDs associated with the text, which can be used for more efficient token-based parsing if supported by the implementation.
     */ 
    void parse(JsonContainer& message, const std::optional<std::vector<int64_t>>& tokens = std::nullopt) override;

    /**
     * @brief Parse incremental text content for Llama 3 JSON tool calls.
     *
     * Current implementation does not perform incremental tool-call extraction and
     * returns the input delta_text unchanged.
     */
    std::string parseChunk(
        JsonContainer& delta_message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override {
        (void)delta_message;
        (void)delta_tokens;
        return delta_text;
    }

    /**
     * @brief Reset internal parser state.
     *
     * Current implementation has no internal state to reset.
     */
    void reset() override {}

private:
    class Llama3JsonToolParserImpl;
    std::unique_ptr<Llama3JsonToolParserImpl> m_impl;
};

}  // namespace genai
}  // namespace ov
