// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "xgrammar_backend.hpp"
#include <iostream>

namespace ov {
namespace genai {

XGrammarStructuredOutput::XGrammarStructuredOutput(const ov::genai::Tokenizer::TokenizerImpl& tokenizer_impl, std::optional<int> vocab_size) {
    auto vocab_vector = tokenizer_impl.m_vocab;
    if (!vocab_size.has_value()) {
        vocab_size = vocab_vector.size();
    }
    
    auto tokenizer_info = xgrammar::TokenizerInfo(
        std::move(vocab_vector),
        xgrammar::VocabType::RAW,
        vocab_size,
        std::vector<int32_t>{static_cast<int32_t>(tokenizer_impl.m_eos_token_id)},
        true
    );
    m_grammar_compiler = std::make_unique<xgrammar::GrammarCompiler>(std::move(tokenizer_info));
}


xgrammar::Grammar XGrammarStructuredOutput::parse_compound_grammar(const StructuredOutputConfig::CompoundGrammar& compound_grammar) {
    return std::visit([](const auto& grammar) -> xgrammar::Grammar {
        using T = std::decay_t<decltype(grammar)>;
        if constexpr (std::is_same_v<T, StructuredOutputConfig::Regex>) {
            return xgrammar::Grammar::FromRegex(grammar.value);
        } else if constexpr (std::is_same_v<T, StructuredOutputConfig::JSONSchema>) {
            return xgrammar::Grammar::FromJSONSchema(grammar.value);
        } else if constexpr (std::is_same_v<T, StructuredOutputConfig::EBNF>) {
            return xgrammar::Grammar::FromEBNF(grammar.value);
        } else if constexpr (std::is_same_v<T, std::shared_ptr<StructuredOutputConfig::Concat>>) {
            return xgrammar::Grammar::Concat({
                XGrammarStructuredOutput::parse_compound_grammar(grammar->left),
                XGrammarStructuredOutput::parse_compound_grammar(grammar->right)
            });
        } else if constexpr (std::is_same_v<T, std::shared_ptr<StructuredOutputConfig::Union>>) {
            return xgrammar::Grammar::Union({
                XGrammarStructuredOutput::parse_compound_grammar(grammar->left),
                XGrammarStructuredOutput::parse_compound_grammar(grammar->right)
            });
        } else {
            OPENVINO_THROW(
                "Cannot compile the compound grammar. Unsupported compound grammar type. "
                "Supported types are: Regex, JSONSchema, EBNF, Union, Concat."
            );
        }
    }, compound_grammar);
}

xgrammar::Grammar XGrammarStructuredOutput::create_grammar(const std::optional<StructuredOutputConfig>& structured_output_config) {
    // Default constructor for xgrammar::Grammar is not enabled,
    // create explicitly an empty grammar.
    xgrammar::Grammar grammar = xgrammar::Grammar::FromEBNF("root ::= root");
    if (!structured_output_config.has_value()) {
        return grammar;
    }

    if (structured_output_config.value().json_schema.has_value()) {
        grammar = xgrammar::Grammar::FromJSONSchema(structured_output_config.value().json_schema.value());
    } else if (structured_output_config.value().regex.has_value()) {
        grammar = xgrammar::Grammar::FromRegex(structured_output_config.value().regex.value());
    } else if (structured_output_config.value().grammar.has_value()) {
        grammar = xgrammar::Grammar::FromEBNF(structured_output_config.value().grammar.value());
    } else if (structured_output_config.value().structural_tags_config.has_value()) {
        std::vector<xgrammar::StructuralTagItem> xgrammar_structural_tags;
        for (const auto& tag : structured_output_config.value().structural_tags_config.value().structural_tags) {
            auto structural_tag = xgrammar::StructuralTagItem{tag.begin, tag.schema, tag.end};
            xgrammar_structural_tags.push_back(std::move(structural_tag));
        }
        grammar = xgrammar::Grammar::FromStructuralTag(
            xgrammar_structural_tags, structured_output_config.value().structural_tags_config.value().triggers
        );
    } else if (structured_output_config.compound_grammar.has_value()) {
        grammar = parse_compound_grammar(*structured_output_config.compound_grammar);
    } else {
        OPENVINO_THROW("No grammar definition provided for structured output generation.");
    }
    return grammar;
}

void XGrammarStructuredOutput::validate_grammar(const std::optional<StructuredOutputConfig>& structured_output_config) {
    create_grammar(structured_output_config);
}

std::shared_ptr<LogitTransformers::ILogitTransformer>
XGrammarStructuredOutput::get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters) {
    if (!sampling_parameters.structured_output_config.has_value()) {
        OPENVINO_THROW("Structured output is not enabled in the provided GenerationConfig.");
    }
    sampling_parameters.structured_output_config.value().validate();
    auto grammar = create_grammar(sampling_parameters.structured_output_config);
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
): m_grammar_matcher(
      compiled_grammar,
      override_stop_tokens,
      terminate_without_stop_token,
      max_rollback_tokens
  ) {
    m_vocab_size = compiled_grammar.GetTokenizerInfo().GetVocabSize();
    
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
