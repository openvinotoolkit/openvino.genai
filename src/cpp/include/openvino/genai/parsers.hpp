// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/genai/text_streamer.hpp"

namespace ov {
namespace genai {

enum class ParsingState {
    CONTENT,
    REASONING,
    TOOL_CALLING,
    UNDEFINED
};


using ParsedMessage = std::map<std::string, std::string>;

class ParsedJSONMessage {
public:
    std::map<std::string, std::string> content;
};


// struct DeltaMessage {
//     std::map<std::string, std::string> content;
//     std::optional<std::string> content;
//     std::optional<std::string> reasoning_content;
//     ParsingState state = ParsingState::UNDEFINED;
    
//     // std::vector<DeltaToolCall> tool_calls;

//     DeltaMessage() = default;
// };


class IncrementalParserBase {
public:
    IncrementalParserBase() = default;

    virtual ParsedMessage parse(
        const std::string& previous_text, 
        const std::string& delta_text, 
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) = 0;
};

class ParserBase {
public:
    ParserBase() = default;

    virtual ParsedMessage parse(ParsedMessage& text) = 0;
};



class TextParserStreamer : public ov::genai::TextStreamer {
public:
    TextParserStreamer(const Tokenizer& tokenizer);

    virtual StreamingStatus write(ParsedMessage& message) = 0;

    ov::genai::CallbackTypeVariant write(std::string message);
private:
    std::string m_text_buffer;
    std::shared_ptr<IncrementalParserBase> m_reasoning_parser;
    std::shared_ptr<ParserBase> m_tool_calling_parser;
};

class Llama32PythonicParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    Llama32PythonicParser(bool keep_original_content = true) : m_keep_original_content(keep_original_content) {}

    ParsedMessage parse(ParsedMessage& input) override;

private:
    bool m_keep_original_content = true;
};

class BaseReasoningParser : public ParserBase{
public:
    BaseReasoningParser(bool expect_open_tag = true, bool keep_original_content = true, std::string open_tag = "<think>", std::string close_tag = "</think>") :
    m_expect_open_tag(expect_open_tag), 
    m_keep_original_content(keep_original_content),
    m_open_tag(open_tag), 
    m_close_tag(close_tag) {}

    ParsedMessage parse(ParsedMessage& input) override;

private:
    bool m_expect_open_tag = true;
    bool m_keep_original_content = true;
    std::string m_open_tag = "<think>";
    std::string m_close_tag = "<think/>";
};


}  // namespace genai
}  // namespace ov
