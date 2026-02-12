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
     */
    virtual void parse(JsonContainer& message) = 0;
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
    
    /**
     * @brief Parse complete text content at the end of generate call.
     *
     * This method processes the entire text content and extracts or modifies
     * information as needed. The results are stored in the provided JsonContainer.
     *
     * @param message JsonContainer containing the text to parse and to store results
     */
    void parse(JsonContainer& message) override;
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
     */
    void parse(JsonContainer& message) override;
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
     */
    void parse(JsonContainer& message) override;
private:
    class Llama3JsonToolParserImpl;
    std::unique_ptr<Llama3JsonToolParserImpl> m_impl;
};

/**
 * @brief Abstract base class for incremental parsers that process text during streaming.
 *
 * Derived classes must implement both the `parse()` and `reset()` methods, as these are pure virtual.
 *
 * Use `IncrementalParser` when you need to process text as it is generated (e.g., in streaming scenarios),
 * handling partial content and maintaining internal state between increments. 
 * In case of processing complete text after generation has finished, `Parser` should be used.
 *
 * Example:
 * @code
 * class MyIncrementalParser : public ov::genai::IncrementalParser {
 * public:
 *     std::string parse(JsonContainer& delta_message, std::string& delta_text,
 *                       const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt) override {
 *         // Implement incremental parsing logic here
 *         return delta_text; // Example: simply return the input
 *     }
 *     void reset() override {
 *         // Reset internal state here
 *     }
 * };
 * @endcode
 */
class OPENVINO_GENAI_EXPORTS IncrementalParser {
public:
    IncrementalParser() = default;

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
    virtual std::string parse(
        JsonContainer& delta_message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) = 0;

    /**
     * @brief Reset the internal state of the parser.
     */
    virtual void reset() = 0;

    virtual ~IncrementalParser() = default;
};

/**
 * @brief Incremental parser for reasoning content with configurable tags.
 *
 * Extracts text with open and close tags. Original JsonContainer must have 'content' field.
 * The reasoning content is stored in the 'reasoning_content' field of the JsonContainer.
 */
class OPENVINO_GENAI_EXPORTS ReasoningIncrementalParser : public IncrementalParser {
public:
    /**
     * @brief Constructor for ReasoningIncrementalParser.
     *
     * @param expect_open_tag If true then open_tag is expected to be generated, if false then it's already part of the model input string
     * @param keep_original_content If true then original 'content' is preserved, otherwise reasoning text is removed from 'content'
     * @param open_tag The opening tag (default: "<think>")
     * @param close_tag The closing tag (default: "</think>")
     */
    ReasoningIncrementalParser(bool expect_open_tag = true,
                    bool keep_original_content = true,
                    const std::string& open_tag = "<think>",
                    const std::string& close_tag = "</think>");
    virtual ~ReasoningIncrementalParser();

    /**
     * @brief Parse reasoning content incrementally.
     *
     * Processes text streams containing reasoning sections marked by configurable tags.
     * Can filter out reasoning content or preserve it based on parser configuration.
     *
     * @param delta_message JsonContainer to store parsed results and reasoning metadata
     * @param delta_text New text chunk to be processed in this step
     * @param delta_tokens Optional vector of new token IDs to be processed
     * @return std::string filtered text that should be added to the 'content'
     */
    std::string parse(
        JsonContainer& delta_message,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override;

    /**
     * @brief Reset the internal state of the parser.
     */
    void reset() override;
private:
    class ReasoningParserImpl;
    std::unique_ptr<ReasoningParserImpl> m_impl;
};

/**
 * @brief Incremental parser for DeepSeek R1 model reasoning format.
 *
 * DeepSeekR1ReasoningIncrementalParser is configured for the DeepSeek R1 model's reasoning format, which doesn't expect an opening tag.
 */
class OPENVINO_GENAI_EXPORTS DeepSeekR1ReasoningIncrementalParser : public ReasoningIncrementalParser {
public:
    DeepSeekR1ReasoningIncrementalParser() : ReasoningIncrementalParser(/*expect_open_tag=*/false) {};
};

/**
 * @brief Incremental parser for Phi-4 model reasoning format.
 *
 * Phi4ReasoningIncrementalParser is configured specifically for the Phi-4 model's reasoning format, which expects an opening tag by default.
 */
class OPENVINO_GENAI_EXPORTS Phi4ReasoningIncrementalParser : public ReasoningIncrementalParser {
public:
    Phi4ReasoningIncrementalParser() : ReasoningIncrementalParser(/*expect_open_tag=*/true) {};
};

}  // namespace genai
}  // namespace ov
