// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "xgrammar_backend.hpp"
#include <iostream>

namespace ov {
namespace genai {

XGrammarStructuredOutput::XGrammarStructuredOutput(const Tokenizer& tokenizer, std::optional<int> vocab_size) {
    auto vocab_vector = tokenizer.get_vocab_vector();
    if (!vocab_size.has_value()) {
        vocab_size = vocab_vector.size();
    }
    
    auto tokenizer_info = xgrammar::TokenizerInfo(
        std::move(vocab_vector),
        xgrammar::VocabType::RAW,
        vocab_size,
        std::vector<int32_t>{static_cast<int32_t>(tokenizer.get_eos_token_id())},
        true
    );
    m_grammar_compiler = std::make_unique<xgrammar::GrammarCompiler>(std::move(tokenizer_info));
}

std::shared_ptr<LogitTransformers::ILogitTransformer>
XGrammarStructuredOutput::get_logits_transformer(const GenerationConfig& sampling_parameters) {
    OPENVINO_ASSERT(sampling_parameters.is_structured_output_generation(),
                   "XGrammarStructuredOutput can only be used for structured output generation");
    
    auto& guided_gen_config = *sampling_parameters.structured_output_config;
    guided_gen_config.validate();

    xgrammar::Grammar grammar;
    if (guided_gen_config.json_schema.has_value()) {
        // std::cout << *guided_generation_config.json_schema << std::endl;
        grammar = xgrammar::Grammar::FromJSONSchema(*guided_gen_config.json_schema);
    } else if (guided_gen_config.regex.has_value()) {
        grammar = xgrammar::Grammar::FromRegex(*guided_gen_config.regex);
    } else if (guided_gen_config.grammar.has_value()) {
        grammar = xgrammar::Grammar::FromEBNF(*guided_gen_config.grammar);
    }

    auto compiled_grammar = m_grammar_compiler->CompileGrammar(grammar);
    std::vector<int> override_stop_tokens(sampling_parameters.stop_token_ids.begin(), sampling_parameters.stop_token_ids.end());
    
    return std::make_shared<LogitTransformers::XGrammarLogitsTransformer>(std::move(compiled_grammar), override_stop_tokens);
}

namespace LogitTransformers {

XGrammarLogitsTransformer::XGrammarLogitsTransformer(
    const xgrammar::CompiledGrammar& compiled_grammar,
    std::optional<std::vector<int>> override_stop_tokens,
    bool terminate_without_stop_token,
    int max_rollback_tokens
) {
    m_vocab_size = compiled_grammar.GetTokenizerInfo().GetVocabSize();
    m_grammar_matcher = xgrammar::GrammarMatcher(
        compiled_grammar,
        override_stop_tokens,
        terminate_without_stop_token,
        max_rollback_tokens
    );
    
    // Divide vocab into 32 for bitmask and ceil to the nearest integer
    // This is to ensure that we can use a bitmask to represent the vocabulary
    const size_t bitmask_size = std::ceil(static_cast<float>(m_vocab_size) / 32.0f); 
    m_token_bitmask_ov = ov::Tensor(ov::element::i32, {bitmask_size});
    std::fill(m_token_bitmask_ov.data<int32_t>(), m_token_bitmask_ov.data<int32_t>() + bitmask_size, -1);

    for (size_t i = 0; i < bitmask_size; ++i) {
        m_token_bitmask_ov.data<int32_t>()[i] = -1;
    }
    m_token_bitmask = std::make_shared<DLTensor>();
    m_token_bitmask->data = m_token_bitmask_ov.data<int32_t>();
    m_token_bitmask->device = DLDevice{kDLCPU, 0};
    m_token_bitmask->ndim = 1;
    m_token_bitmask->dtype = DLDataType{kDLInt, 32, 1};
    m_token_bitmask->byte_offset = 0;
    m_token_bitmask->strides = nullptr; // No strides, tensor is compact
    m_bitmask_shape = {static_cast<int64_t>(m_token_bitmask_ov.get_size())};
    m_token_bitmask->shape = &m_bitmask_shape[0];
    
    m_next_token_logits = std::make_shared<DLTensor>();
    m_next_token_logits->device = DLDevice{kDLCPU, 0};
    m_next_token_logits->ndim = 1;
    m_next_token_logits->dtype = DLDataType{kDLFloat, 32, 1};
    m_next_token_logits->byte_offset = 0;
    m_next_token_logits->strides = nullptr; // No strides, tensor is compact
    m_logits_shape = {static_cast<int64_t>(m_vocab_size)};
    m_next_token_logits->shape = &m_logits_shape[0];
}

void XGrammarLogitsTransformer::accept_tokens(const TokenIds& input_ids) {
    for (const auto& token : input_ids) {
        m_grammar_matcher.AcceptToken(token);
    }
}

void XGrammarLogitsTransformer::apply(Logits& logits) {
    m_next_token_logits->data = logits.m_data;

    m_grammar_matcher.FillNextTokenBitmask(m_token_bitmask.get());
    if (!m_grammar_matcher.IsTerminated()) {
        xgrammar::ApplyTokenBitmaskInplaceCPU(m_next_token_logits.get(), *m_token_bitmask, m_vocab_size);
    }
}

} // namespace LogitTransformers

} // namespace genai
} // namespace ov
