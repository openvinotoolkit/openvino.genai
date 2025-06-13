// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <xgrammar/xgrammar.h>
#include <xgrammar/compiler.h>
#include <xgrammar/tokenizer_info.h>
#include "structured_output_controller.hpp"
#include "dlpack/dlpack.h"
#include <memory>
#include <optional>
#include <vector>

namespace ov {
namespace genai {

namespace LogitTransformers {

class XGrammarLogitsTransformer : public IStatefulLogitTransformer {
public:                            
    XGrammarLogitsTransformer(
        const xgrammar::CompiledGrammar& compiled_grammar,
        std::optional<std::vector<int>> override_stop_tokens = std::nullopt,
        bool terminate_without_stop_token = false,
        int max_rollback_tokens = 0
    );

    void accept_tokens(const TokenIds& input_ids) override;

    void apply(Logits& logits) override;
protected:
    xgrammar::GrammarMatcher m_grammar_matcher;

    ov::Tensor m_token_bitmask_ov;
    std::shared_ptr<DLTensor> m_token_bitmask;
    std::shared_ptr<DLTensor> m_next_token_logits;
    std::vector<int64_t> m_logits_shape;
    std::vector<int64_t> m_bitmask_shape;
    int m_vocab_size;
};

} // namespace LogitTransformers



class XGrammarStructuredOutput : public IStructuredOutputImpl {
public:
    XGrammarStructuredOutput(const Tokenizer& tokenizer, std::optional<int> vocab_size = std::nullopt);    
    std::shared_ptr<LogitTransformers::ILogitTransformer> get_logits_transformer(const GenerationConfig& sampling_parameters) override;
private:
    std::unique_ptr<xgrammar::GrammarCompiler> m_grammar_compiler;
};


// Static initializer for XGrammar backend registration
static bool registerXGrammarBackend() {
    StructuredOutputController::register_backend("xgrammar",
        [](const ov::genai::Tokenizer& tokenizer, std::optional<int> vocab_size) {
            return std::make_unique<XGrammarStructuredOutput>(tokenizer, vocab_size);
        });
        return true;
}

// This ensures the function is called during static initialization
static bool xgrammar_registered = registerXGrammarBackend();

} // namespace genai
} // namespace ov
