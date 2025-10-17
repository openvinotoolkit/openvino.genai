// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <memory>
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
};

class ReasoningParser : public IncrementalParserBase {
private:
    class ReasoningParserImpl;
    std::shared_ptr<ReasoningParserImpl> m_impl;
public:
    ReasoningParser(bool expect_open_tag = true,
                    bool keep_original_content = true, 
                    std::string open_tag="<think>", 
                    std::string close_tag="</think>");

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
};

class Phi4ReasoningParser : public ReasoningParser {
public:
    explicit Phi4ReasoningParser(bool expect_open_tag = false) : ReasoningParser(expect_open_tag) {};
};

class ParserBase {
public:
    ParserBase() = default;
    virtual void parse(JsonContainer& text) = 0;
};

class Llama32PythonicToolParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    explicit Llama32PythonicToolParser(bool keep_original_content = true);
    void parse(JsonContainer& input) override;
private:
    class Llama32PythonicToolParserImpl;
    std::shared_ptr<Llama32PythonicToolParserImpl> m_impl;
};

class Llama32JsonToolParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    explicit Llama32JsonToolParser(bool keep_original_content = true);
    void parse(JsonContainer& input) override;
private:
    class Llama32JsonToolParserImpl;
    std::shared_ptr<Llama32JsonToolParserImpl> m_impl;
};

class BaseReasoningParser : public ParserBase{
public:
    BaseReasoningParser(bool expect_open_tag = true, bool keep_original_content = true, std::string open_tag = "<think>", std::string close_tag = "</think>");
    void parse(JsonContainer& input) override;
private:
    class BaseReasoningParserImpl;
    std::shared_ptr<BaseReasoningParserImpl> m_impl;
};

}  // namespace genai
}  // namespace ov
