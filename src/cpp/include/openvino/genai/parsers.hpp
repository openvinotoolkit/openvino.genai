// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <memory>
#include <vector>
#include "openvino/genai/json_container.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS IncrementalParser {
public:
    IncrementalParser() = default;

    // We return string which with filtered text to be added to content.
    virtual std::string parse(
        JsonContainer& message,
        const std::string& previous_text, 
        std::string& delta_text, 
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) = 0;

    virtual ~IncrementalParser() = default;
};

class OPENVINO_GENAI_EXPORTS ReasoningIncrementalParser : public IncrementalParser {
private:
    class ReasoningParserImpl;
    std::unique_ptr<ReasoningParserImpl> m_impl;
public:
    ReasoningIncrementalParser(bool expect_open_tag = true,
                    bool keep_original_content = true, 
                    const std::string& open_tag = "<think>", 
                    const std::string& close_tag = "</think>");
    virtual ~ReasoningIncrementalParser();

    std::string parse(
        JsonContainer& message,
        const std::string& previous_text, 
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override;
};

class OPENVINO_GENAI_EXPORTS DeepSeekR1ReasoningIncrementalParser : public ReasoningIncrementalParser {
public:
    explicit DeepSeekR1ReasoningIncrementalParser(bool expect_open_tag = false) : ReasoningIncrementalParser(expect_open_tag) {};
};

class OPENVINO_GENAI_EXPORTS Phi4ReasoningIncrementalParser : public ReasoningIncrementalParser {
public:
    explicit Phi4ReasoningIncrementalParser(bool expect_open_tag = true) : ReasoningIncrementalParser(expect_open_tag) {};
};

class OPENVINO_GENAI_EXPORTS Parser {
public:
    Parser() = default;
    virtual ~Parser();
    virtual void parse(JsonContainer& text) = 0;
};

class OPENVINO_GENAI_EXPORTS ReasoningParser : public Parser {
public:
    ReasoningParser(
        bool expect_open_tag = true, 
        bool keep_original_content = true, 
        const std::string& open_tag = "<think>", 
        const std::string& close_tag = "</think>");
    void parse(JsonContainer& message) override;
    ~ReasoningParser();
private:
    class ReasoningParserImpl;
    std::unique_ptr<ReasoningParserImpl> m_impl;
};

class OPENVINO_GENAI_EXPORTS Llama3PythonicToolParser : public Parser {
// Does not modify original content, only extracts and adds tool calls
public:
    explicit Llama3PythonicToolParser(bool keep_original_content = true);
    ~Llama3PythonicToolParser();
    void parse(JsonContainer& message) override;
private:
    class Llama3PythonicToolParserImpl;
    std::unique_ptr<Llama3PythonicToolParserImpl> m_impl;
};

class OPENVINO_GENAI_EXPORTS Llama3JsonToolParser : public Parser {
// Does not modify original content, only extracts and adds tool calls
public:
    explicit Llama3JsonToolParser(bool keep_original_content = true);
    ~Llama3JsonToolParser();
    void parse(JsonContainer& message) override;
private:
    class Llama3JsonToolParserImpl;
    std::unique_ptr<Llama3JsonToolParserImpl> m_impl;
};

}  // namespace genai
}  // namespace ov
