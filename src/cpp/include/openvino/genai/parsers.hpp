// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <memory>
#include <vector>
#include "openvino/genai/json_container.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS IncrementalParserBase {
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

    virtual ~IncrementalParserBase() = default;
};

class OPENVINO_GENAI_EXPORTS ReasoningParser : public IncrementalParserBase {
private:
    class ReasoningParserImpl;
    std::unique_ptr<ReasoningParserImpl> m_impl;
public:
    ReasoningParser(bool expect_open_tag = true,
                    bool keep_original_content = true, 
                    const std::string& open_tag = "<think>", 
                    const std::string& close_tag = "</think>");
    virtual ~ReasoningParser();

    std::string parse(
        JsonContainer& msg,
        const std::string& previous_text, 
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override;
};

class OPENVINO_GENAI_EXPORTS DeepSeekR1ReasoningParser : public ReasoningParser {
public:
    explicit DeepSeekR1ReasoningParser(bool expect_open_tag = false) : ReasoningParser(expect_open_tag) {};
};

class OPENVINO_GENAI_EXPORTS Phi4ReasoningParser : public ReasoningParser {
public:
    explicit Phi4ReasoningParser(bool expect_open_tag = true) : ReasoningParser(expect_open_tag) {};
};

class ParserBase {
public:
    ParserBase() = default;
    virtual ~ParserBase();
    virtual void parse(JsonContainer& text) = 0;
};

class OPENVINO_GENAI_EXPORTS Llama3PythonicToolParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    explicit Llama3PythonicToolParser(bool keep_original_content = true);
    ~Llama3PythonicToolParser();
    void parse(JsonContainer& input) override;
private:
    class Llama3PythonicToolParserImpl;
    std::unique_ptr<Llama3PythonicToolParserImpl> m_impl;
};

class OPENVINO_GENAI_EXPORTS Llama3JsonToolParser : public ParserBase {
// Does not modify original content, only extracts and adds tool calls
public:
    explicit Llama3JsonToolParser(bool keep_original_content = true);
    ~Llama3JsonToolParser();
    void parse(JsonContainer& input) override;
private:
    class Llama3JsonToolParserImpl;
    std::unique_ptr<Llama3JsonToolParserImpl> m_impl;
};

class OPENVINO_GENAI_EXPORTS BaseReasoningParser : public ParserBase{
public:
    BaseReasoningParser(
        bool expect_open_tag = true, 
        bool keep_original_content = true, 
        const std::string& open_tag = "<think>", 
        const std::string& close_tag = "</think>");
    void parse(JsonContainer& input) override;
    ~BaseReasoningParser();
private:
    class BaseReasoningParserImpl;
    std::unique_ptr<BaseReasoningParserImpl> m_impl;
};

// TODO: DeepSeekR1ReasoningParser -> DeepSeekR1IncrementalParser

}  // namespace genai
}  // namespace ov
