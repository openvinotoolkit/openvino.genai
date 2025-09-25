// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <memory>
#include <variant>
#include <map>
#include <optional>
#include <vector>

namespace ov {
namespace genai {


using ParsedMessage = std::map<std::string, std::string>;

class IncrementalParserBase {
public:
    IncrementalParserBase() = default;

    virtual ParsedMessage parse(
        ParsedMessage& msg,
        const std::string& previous_text, 
        const std::string& delta_text, 
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) = 0;

    virtual bool is_active() const = 0;
    static std::map<std::string, std::shared_ptr<IncrementalParserBase>> registered_parsers;
};

class DeepSeekR1ReasoningParser : public IncrementalParserBase {
private:
    bool m_starts_with_thinking = true;
    bool m_think_tag_opened = false;
    bool m_deactivated = false;
    std::string m_open_tag = "<think>";
    std::string m_close_tag = "</think>";
public:
    DeepSeekR1ReasoningParser(bool starts_with_thinking = true) : m_starts_with_thinking(starts_with_thinking) {};
    std::map<std::string, std::string> accumulated_parsed;

    ParsedMessage parse(
        ParsedMessage& msg,
        const std::string& previous_text, 
        const std::string& delta_text,
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override;
    static std::string name() { return "DeepSeekR1ReasoningParser"; }
    bool is_active() const override;
};

class ParserBase {
public:
    ParserBase() = default;

    virtual ParsedMessage parse(ParsedMessage& text) = 0;
    static std::map<std::string, std::shared_ptr<ParserBase>> registered_parsers;
};

using ParserVariant = std::variant<std::shared_ptr<IncrementalParserBase>, std::string>;

class Llama32PythonicParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    Llama32PythonicParser(bool keep_original_content = true) : m_keep_original_content(keep_original_content) {}

    ParsedMessage parse(ParsedMessage& input) override;
    static std::string name() { return "Llama32PythonicParser"; }
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
    std::string m_close_tag = "</think>";
};


}  // namespace genai
}  // namespace ov
