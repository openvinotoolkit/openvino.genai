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

    static std::shared_ptr<IncrementalParserBase> get_parser(std::string name);
};

class ReasoningParser : public IncrementalParserBase {
private:
    class ReasoningParserImpl;
    std::shared_ptr<ReasoningParserImpl> m_impl;
public:
    ReasoningParser(bool expect_open_tag = true,
                    bool keep_original_content = true);

    std::string parse(
        JsonContainer& msg,
        const std::string& previous_text, 
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override;
};

class DeepSeekR1ReasoningParser : public ReasoningParser {
public:
    explicit DeepSeekR1ReasoningParser(bool expect_open_tag = true) : ReasoningParser(expect_open_tag) {};
    static std::string name() { return "DeepSeekR1ReasoningParser"; }
};

class Phi4ReasoningParser : public ReasoningParser {
public:
    explicit Phi4ReasoningParser(bool expect_open_tag = false) : ReasoningParser(expect_open_tag) {};
    static std::string name() { return "Phi4ReasoningParser"; }
};

class ParserBase {
public:
    ParserBase() = default;
    virtual JsonContainer parse(JsonContainer& text) = 0;
    static std::shared_ptr<ParserBase> get_parser(std::string name);
};

class Llama32PythonicToolParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    // TODO: Check that vLLM has the same default.
    explicit Llama32PythonicToolParser(bool keep_original_content = true) : m_keep_original_content(keep_original_content) {}

    JsonContainer parse(JsonContainer& input) override;
    static std::string name() { return "Llama32PythonicToolParser"; }
private:
    bool m_keep_original_content;
};

class Llama32JsonToolParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    // TODO: Check that vLLM has the same default.
    explicit Llama32JsonToolParser(bool keep_original_content = true) : m_keep_original_content(keep_original_content) {}

    JsonContainer parse(JsonContainer& input) override;
    static std::string name() { return "Llama32JsonToolParser"; }
private:
    bool m_keep_original_content;
};

class BaseReasoningParser : public ParserBase{
public:
    BaseReasoningParser(bool expect_open_tag = true, bool keep_original_content = true, std::string open_tag = "<think>", std::string close_tag = "</think>");
    JsonContainer parse(JsonContainer& input) override;
private:
    class BaseReasoningParserImpl;
    std::shared_ptr<BaseReasoningParserImpl> m_impl;
};

}  // namespace genai
}  // namespace ov
