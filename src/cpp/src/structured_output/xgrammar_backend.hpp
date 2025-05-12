// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <xgrammar/xgrammar.h>
#include <xgrammar/compiler.h>
#include <xgrammar/tokenizer_info.h>
#include "structured_output_controller.hpp"


namespace ov {
namespace genai {

namespace LogitTransformers {

class ILogitTransformer;
//class ILogitTransformer {
//public:
//    virtual void apply(Logits& logits) = 0;
//
//    virtual bool is_applicable(size_t generated_tokens_cnt = 0) {
//        return true;
//    }
//};


class XGrammarLogitsTransformer : public IStructuredOutputBaseLogitTransformer {
public:
    XGrammarLogitsTransformer(const CompiledGrammar& compiled_grammar,
                              std::optional<std::vector<int>> override_stop_tokens = std::nullopt,
                              bool terminate_without_stop_token = false,
                              int max_rollback_tokens = 0) {
        m_grammar_matcher = xgrammar::GrammarMatcher(compiled_grammar,
                                                    override_stop_tokens,
                                                    terminate_without_stop_token,
                                                    max_rollback_tokens);
    }
    void accept_tokens(const TokenIds& input_ids) {
        for (const auto& token : input_ids) {
            m_grammar_matcher.AcceptToken(token);
        }
    };
protected:
    xgrammar::GrammarMatcher m_grammar_matcher;
}
} // namespace LogitTransformers


class XGrammarStructuredOutput : public IStructuredOutputBaseImpl {
public:
    XGrammarStructuredOutput(const Tokenizer& tokenizer, std::optional<int> vocab_size = std::nullopt) {
        auto tokenizer_info = xgrammar::TokenizerInfo(
            tokenizer.get_vocab_vector(),
            xgrammar::VocabType::RAW,  // VocabType vocab_type
            vocab_size,  // int vocab_size
            std::nullopt,  // std::optional<std::vector<int32_t>> stop_token_ids
            false  // bool add_prefix_space
        );
        m_grammar_compiler = std::make_unique<xgrammar::GrammarCompiler>(std::move(tokenizer_info));
    }

    LogitTransformers::XGrammarLogitsTransformer get_json_schema_logits_transformer(const std::string& json_schema) {
        auto compiled_grammar = m_grammar_compiler->CompileJSONSchema(json_schema);
        return XGrammarLogitsTransformer(compiled_grammar);
    };

private:
    std::unique_ptr<xgrammar::GrammarCompiler> m_grammar_compiler;
};


} // namespace genai
}// namespace ov
