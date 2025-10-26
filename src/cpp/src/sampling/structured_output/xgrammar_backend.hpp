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

/**
 * @brief Logit transformer that applies XGrammar grammar matching to logits.
 * 
 * It encapsulates the XGrammar backend implementation and exposes a public apply method, 
 * which applies grammar constraints to the logits each time a new token is generated, and 
 * accepts tokens to update the internal state of the grammar matcher.
 */
class XGrammarLogitsTransformer : public IStatefulLogitTransformer {
public:                            
    explicit XGrammarLogitsTransformer(
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
    std::vector<int64_t> m_logits_strides = {1};
    std::vector<int64_t> m_bitmask_shape;
    std::vector<int64_t> m_bitmask_strides = {1};
    int m_vocab_size;
};

} // namespace LogitTransformers


/**
 * @brief XGrammarStructuredOutput is a structured output implementation that uses the XGrammar backend.
 *
 * Inherits from IStructuredOutputImpl and acts as the logit transformer builder for the XGrammar backend.
 */
class XGrammarStructuredOutput : public IStructuredOutputImpl {
public:
    /**
     * @brief Constructs an XGrammarStructuredOutput instance with the given tokenizer and optional vocabulary size.
     *
     * This constructor initializes m_grammar_compiler with the provided tokenizer, which is used to compile the grammar.
     * Instantiating m_grammar_compiler can be expensive, so it is done only once per instance of XGrammarStructuredOutput.
     * @param tokenizer The tokenizer to be used for grammar compilation.
     * @param vocab_size Optional vocabulary size; if not provided, it will be determined from the tokenizer.
     */
    XGrammarStructuredOutput(const ov::genai::Tokenizer::TokenizerImpl& tokenizer_impl, std::optional<int> vocab_size = std::nullopt);

    /**
     * @brief Returns a logit transformer that applies XGrammar grammar matching to logits.
     *
     * The get_logits_transformer method retrieves the appropriate logit transformer based on
     * the JSON schema, regex, or EBNF grammar provided in the GenerationConfig. Note that although
     * m_grammar_compiler is created only once in the constructor, a new logit transformer is created each time
     * get_logits_transformer is called.
     * @param sampling_parameters The generation configuration parameters that may include JSON schema, regex, or EBNF grammar.
     * @return A shared pointer to the logit transformer that applies XGrammar grammar matching.
     */
    std::shared_ptr<LogitTransformers::ILogitTransformer> get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters) override;
    void validate_grammar(const std::optional<StructuredOutputConfig>& structured_output_config) override;
private:
    std::unique_ptr<xgrammar::GrammarCompiler> m_grammar_compiler;

    static xgrammar::Grammar parse_structural_tag(const StructuredOutputConfig::CompoundGrammar& compound_grammar);
    xgrammar::Grammar create_grammar(const std::optional<StructuredOutputConfig>& structured_output_config);
};


// Static initializer for XGrammar backend registration
static bool registerXGrammarBackend() {
    StructuredOutputController::register_backend("xgrammar",
        [](const ov::genai::Tokenizer::TokenizerImpl& tokenizer_impl, std::optional<int> vocab_size) {
            return std::make_unique<XGrammarStructuredOutput>(tokenizer_impl, vocab_size);
        });
        return true;
}

// This ensures the function is called during static initialization
static bool xgrammar_registered = registerXGrammarBackend();

} // namespace genai
} // namespace ov
