// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <memory>
#include <variant>
#include <map>
#include <functional>
#include <optional>
#include <vector>
#include "openvino/genai/json_container.hpp"

namespace ov {
namespace genai {

class IncrementalParserBase {
public:
    IncrementalParserBase() = default;

    // We return string which with filtered text to be added to content.
    virtual std::string parse(
        JsonContainer& msg,
        const std::string& previous_text, 
        std::string& delta_text, 
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) = 0;

    virtual bool is_active() const = 0;
    static std::shared_ptr<IncrementalParserBase> get_parser(std::string name);
};

// Forward declaration
class ReasoningParserImpl;

class ReasoningParser : public IncrementalParserBase {
private:
    std::shared_ptr<ReasoningParserImpl> m_impl;
public:
    ReasoningParser(bool starts_with_thinking = true,
                    bool keep_original_content = true);

    std::string parse(
        JsonContainer& msg,
        const std::string& previous_text, 
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override;
    bool is_active() const override;
};

class DeepSeekR1ReasoningParser : public ReasoningParser {
public:
    DeepSeekR1ReasoningParser(bool starts_with_thinking = true) : ReasoningParser(starts_with_thinking) {};
    static std::string name() { return "DeepSeekR1ReasoningParser"; }
};

class Phi4ReasoningParser : public ReasoningParser {
public:
    Phi4ReasoningParser(bool starts_with_thinking = false) : ReasoningParser(starts_with_thinking) {};
    static std::string name() { return "Phi4ReasoningParser"; }
};

class ParserBase {
public:
    ParserBase() = default;

    virtual JsonContainer parse(JsonContainer& text) = 0;
    static std::shared_ptr<ParserBase> get_parser(std::string name);
};

using ParserVariant = std::variant<std::shared_ptr<IncrementalParserBase>, std::string>;

class Llama32PythonicToolParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    // TODO: Check that vLLM has the same default.
    Llama32PythonicToolParser(bool keep_original_content = true) : m_keep_original_content(keep_original_content) {}

    JsonContainer parse(JsonContainer& input) override;
    static std::string name() { return "Llama32PythonicToolParser"; }
private:
    bool m_keep_original_content = true;
};

class Llama32JsonToolParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    // TODO: Check that vLLM has the same default.
    Llama32JsonToolParser(bool keep_original_content = true) : m_keep_original_content(keep_original_content) {}

    JsonContainer parse(JsonContainer& input) override;
    static std::string name() { return "Llama32JsonToolParser"; }
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

    JsonContainer parse(JsonContainer& input) override;

private:
    bool m_expect_open_tag = true;
    bool m_keep_original_content = true;
    std::string m_open_tag = "<think>";
    std::string m_close_tag = "</think>";
};


}  // namespace genai
}  // namespace ov
