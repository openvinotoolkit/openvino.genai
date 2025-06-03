// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef ENABLE_XGRAMMAR
#include "xgrammar_backend.hpp"
#include <iostream>

namespace ov {
namespace genai {

namespace LogitTransformers {

XGrammarLogitsTransformer::XGrammarLogitsTransformer(
    const Tokenizer& tokenizer, 
    std::optional<int> vocab_size,
    const GenerationConfig& sampling_parameters,
    std::optional<std::vector<int>> override_stop_tokens,
    bool terminate_without_stop_token,
    int max_rollback_tokens
) {
    auto vocab_vector = tokenizer.get_vocab_vector();
    if (!vocab_size.has_value()) {
        vocab_size = vocab_vector.size();
    }

    auto tokenizer_info = xgrammar::TokenizerInfo(
        std::move(vocab_vector),
        xgrammar::VocabType::RAW,
        vocab_size,
        std::vector<int32_t>{2},
        true
    );
    auto grammar_compiler = std::make_unique<xgrammar::GrammarCompiler>(std::move(tokenizer_info));

    OPENVINO_ASSERT(sampling_parameters.is_structured_output_generation(),
                   "XGrammarStructuredOutput can only be used for structured output generation");
    
    auto& guided_gen_config = *sampling_parameters.structured_output_config;
    guided_gen_config.validate();

    xgrammar::Grammar grammar;
    if (guided_gen_config.json_schema.has_value()) {
        grammar = xgrammar::Grammar::FromJSONSchema(*guided_gen_config.json_schema);
    } else if (guided_gen_config.regex.has_value()) {
        grammar = xgrammar::Grammar::FromRegex(*guided_gen_config.regex);
    } else if (guided_gen_config.choices.has_value()) {
        // todo: check this
        grammar = xgrammar::Grammar::FromStructuralTag(std::vector<xgrammar::StructuralTagItem>{}, *guided_gen_config.choices);
    } else if (guided_gen_config.grammar.has_value()) {
        grammar = xgrammar::Grammar::FromEBNF(*guided_gen_config.grammar);
    }
    
    auto compiled_grammar = grammar_compiler->CompileGrammar(grammar);

    m_vocab_size = compiled_grammar.GetTokenizerInfo().GetVocabSize();
    m_grammar_matcher = xgrammar::GrammarMatcher(compiled_grammar,
                                                 override_stop_tokens,
                                                 terminate_without_stop_token,
                                                 max_rollback_tokens);
    size_t bitmask_size = (m_vocab_size + 31) / 32;
    m_token_bitmask_ov = ov::Tensor(ov::element::i32, {bitmask_size});
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
    
    m_next_token_logits = std::make_shared<DLTensor>();
    m_next_token_logits->device = DLDevice{kDLCPU, 0};
    m_next_token_logits->ndim = 1;
    m_next_token_logits->dtype = DLDataType{kDLFloat, 32, 1};
    m_next_token_logits->byte_offset = 0;
    m_next_token_logits->strides = nullptr; // No strides, tensor is compact
    // pointer and size will be set in apply method

    // Shapes are stored as a pointer to a buffer. It's not convenient to allocate a buffer here,
    // because in that case we would need to keep it alive until the apply method is called and 
    // deallocate at the very end. Moreover, the shape of logits is not known at this point.
    // Therefore we will set the shape in the apply method.
}

void XGrammarLogitsTransformer::accept_tokens(const TokenIds& input_ids) {
    for (const auto& token : input_ids) {
        m_grammar_matcher.AcceptToken(token);
    }
}

void XGrammarLogitsTransformer::apply(Logits& logits) {
    // Shapes of logits cannot be set in CTOR, becaues size is known only during apply.
    m_next_token_logits->data = logits.m_data;
    std::vector<int64_t> logits_shape = {static_cast<int64_t>(logits.m_size)};
    m_next_token_logits->shape = &logits_shape[0];
    
    std::vector<int64_t> bitmask_shape = {static_cast<int64_t>(m_token_bitmask_ov.get_size())};
    m_token_bitmask->shape = &bitmask_shape[0];

    m_grammar_matcher.FillNextTokenBitmask(m_token_bitmask.get());
    xgrammar::ApplyTokenBitmaskInplaceCPU(m_next_token_logits.get(), *m_token_bitmask, m_vocab_size);
}

} // namespace LogitTransformers

} // namespace genai
} // namespace ov

#endif // ENABLE_XGRAMMAR
