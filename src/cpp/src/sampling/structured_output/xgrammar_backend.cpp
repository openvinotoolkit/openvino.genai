// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "xgrammar_backend.hpp"
#include <iostream>

namespace ov {
namespace genai {

namespace LogitTransformers {

XGrammarLogitsTransformer::XGrammarLogitsTransformer(
    const xgrammar::CompiledGrammar& compiled_grammar,
    std::optional<std::vector<int>> override_stop_tokens,
    bool terminate_without_stop_token,
    int max_rollback_tokens)
{
    m_vocab_size = compiled_grammar.GetTokenizerInfo().GetVocabSize();
    m_grammar_matcher = xgrammar::GrammarMatcher(compiled_grammar,
                                                 override_stop_tokens,
                                                 terminate_without_stop_token,
                                                 max_rollback_tokens);
    size_t size = (m_vocab_size + 31) / 32;
    m_token_bitmask_ov = ov::Tensor(ov::element::i32, {1, size});
    for (size_t i = 0; i < size; ++i) {
        m_token_bitmask_ov.data<int32_t>()[i] = -1;
    }
    m_token_bitmask = std::make_shared<DLTensor>();
}

void XGrammarLogitsTransformer::accept_tokens(const TokenIds& input_ids) {
    for (const auto& token : input_ids) {
        m_grammar_matcher.AcceptToken(token);
    }
}

void XGrammarLogitsTransformer::apply(Logits& logits) {
    m_token_bitmask->data = m_token_bitmask_ov.data<int32_t>();
    m_token_bitmask->device = DLDevice{kDLCPU, 0};
    m_token_bitmask->ndim = 1;
    m_token_bitmask->dtype = DLDataType{kDLInt, 32, 1};
    std::vector<int64_t> shape = {static_cast<int64_t>(m_token_bitmask_ov.get_shape()[1])};
    m_token_bitmask->shape = &shape[0];
    std::vector<int64_t> strides = {1};
    m_token_bitmask->byte_offset = 0;

    DLTensor* next_token_logits = new DLTensor();
    next_token_logits->data = logits.m_data;
    next_token_logits->device = DLDevice{kDLCPU, 0};
    next_token_logits->ndim = 1;
    next_token_logits->dtype = DLDataType{kDLFloat, 32, 1};
    std::vector<int64_t> shape_2 = {static_cast<int64_t>(logits.m_size)};
    next_token_logits->shape = &shape_2[0];
    strides = {1};
    next_token_logits->byte_offset = 0;

    m_grammar_matcher.FillNextTokenBitmask(m_token_bitmask.get());
    xgrammar::ApplyTokenBitmaskInplaceCPU(next_token_logits, *m_token_bitmask, m_vocab_size);

    // delete next_token_logits; // If you own this pointer, uncomment to avoid leaks
}

} // namespace LogitTransformers

XGrammarStructuredOutput::XGrammarStructuredOutput(const Tokenizer& tokenizer, std::optional<int> vocab_size) {
    auto vocab_vector = tokenizer.get_vocab_vector();
    if (!vocab_size.has_value()) {
        vocab_size = vocab_vector.size();
    }

    auto tokenizer_info = xgrammar::TokenizerInfo(
        std::move(vocab_vector),
        xgrammar::VocabType::BYTE_FALLBACK,
        vocab_size,
        std::vector<int32_t>{2},
        true
    );
    m_grammar_compiler = std::make_unique<xgrammar::GrammarCompiler>(std::move(tokenizer_info));
}

std::shared_ptr<LogitTransformers::ILogitTransformer>
XGrammarStructuredOutput::get_logits_transformer(const GenerationConfig& sampling_parameters) {
    OPENVINO_ASSERT(sampling_parameters.is_guided_generation(),
                  "XGrammarStructuredOutput can only be used for guided generation");
    
    auto& guided_gen_config = *sampling_parameters.guided_generation_config;
    guided_gen_config.validate();

    xgrammar::Grammar grammar;
    if (guided_gen_config.json_schema.has_value()) {
        // std::cout << *guided_generation_config.json_schema << std::endl;
        grammar = xgrammar::Grammar::FromJSONSchema(*guided_gen_config.json_schema);
    } else if (guided_gen_config.regex.has_value()) {
        grammar = xgrammar::Grammar::FromRegex(*guided_gen_config.regex);
    } else if (guided_gen_config.choices.has_value()) {
        // todo: check this
        grammar = xgrammar::Grammar::FromStructuralTag(std::vector<xgrammar::StructuralTagItem>{}, *guided_gen_config.choices);
    } else if (guided_gen_config.grammar.has_value()) {
        grammar = xgrammar::Grammar::FromEBNF(*guided_gen_config.grammar);
    }
    
    auto compiled_grammar = m_grammar_compiler->CompileGrammar(grammar);

    return std::make_shared<LogitTransformers::XGrammarLogitsTransformer>(std::move(compiled_grammar));
}

} // namespace genai
} // namespace ov