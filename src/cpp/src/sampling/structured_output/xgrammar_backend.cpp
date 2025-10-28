// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "xgrammar_backend.hpp"
#include "logger.hpp"
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


xgrammar::Grammar XGrammarStructuredOutput::parse_structural_tag(const StructuredOutputConfig::CompoundGrammar& compound_grammar) {
    
    std::ostringstream oss;

    // compound grammar is already a string JSON representation
    if (std::holds_alternative<std::string>(compound_grammar)) {
        oss << std::get<std::string>(compound_grammar);
    } else {
        oss << "{\"type\": \"structural_tag\", \"format\": ";
        oss << std::visit([](const auto& grammar) -> std::string {
            return StructuredOutputConfig::structural_tag_to_json(grammar);
        }, compound_grammar);
        oss << "}";
    };
    auto result = xgrammar::Grammar::FromStructuralTag(oss.str());
    if (std::holds_alternative<xgrammar::Grammar>(result)) {
        return std::get<xgrammar::Grammar>(result);
    } else {
        const auto& error = std::get<xgrammar::StructuralTagError>(result);
        std::string error_message;
        std::visit([&error_message](const auto& err) {
            if constexpr (std::is_member_function_pointer<decltype(&std::decay_t<decltype(err)>::what)>::value) {
                error_message = err.what();
            } else {
                error_message = "Unknown error type";
            }
        }, error);
        OPENVINO_THROW("Failed to create grammar from structural tag: " + error_message);
    }
}

xgrammar::Grammar XGrammarStructuredOutput::create_grammar(const std::optional<StructuredOutputConfig>& structured_output_config) {
    if (!structured_output_config.has_value()) {
        return xgrammar::Grammar::FromEBNF("root ::= root");
    }

    if (structured_output_config.value().json_schema.has_value()) {
        return xgrammar::Grammar::FromJSONSchema(structured_output_config.value().json_schema.value());
    } else if (structured_output_config.value().regex.has_value()) {
        return xgrammar::Grammar::FromRegex(structured_output_config.value().regex.value());
    } else if (structured_output_config.value().grammar.has_value()) {
        return xgrammar::Grammar::FromEBNF(structured_output_config.value().grammar.value());
    } else if (structured_output_config.value().structural_tags_config.has_value()) {
        return std::visit([](const auto& config) -> xgrammar::Grammar {
            using ConfigType = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<ConfigType, StructuralTagsConfig>) {
                // Old format: StructuralTagsConfig
                Logger::warn(
                    "The use of \"structural_tags_config\" with StructuralTagsConfig instance is deprecated and will be removed in future releases. "
                    "Use TriggeredTags instead."
                );
                
                std::ostringstream oss;
                oss << "{\"type\": \"structural_tag\", \"format\": " << config.to_json() << "}";
                auto result = xgrammar::Grammar::FromStructuralTag(oss.str());
                if (std::holds_alternative<xgrammar::Grammar>(result)) {
                    return std::get<xgrammar::Grammar>(result);
                } else {
                    const auto& error = std::get<xgrammar::StructuralTagError>(result);
                    std::string error_message;
                    std::visit([&error_message](const auto& err) {
                        if constexpr (std::is_member_function_pointer<decltype(&std::decay_t<decltype(err)>::what)>::value) {
                            error_message = err.what();
                        } else {
                            error_message = "Unknown error type";
                        }
                    }, error);
                    OPENVINO_THROW("Failed to create grammar from structural tag: " + error_message);
                }
            } else {
                // New format: StructuralTag
                return parse_structural_tag(config);
            }
        }, structured_output_config.value().structural_tags_config.value());
    } else if (structured_output_config.value().compound_grammar.has_value()) {
        Logger::warn(
            "The use of \"compound_grammar\" is deprecated and will be removed in future releases.\n" 
            "Pass the same input to \"structural_tags_config\" instead."
        );
        return parse_structural_tag(structured_output_config.value().compound_grammar.value());
    }

    OPENVINO_THROW("No grammar definition provided for structured output generation.");
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
    m_token_bitmask->strides = &m_bitmask_strides[0];  // xgrammar expects strides to be set, even for compact tensors
    m_bitmask_shape = {static_cast<int64_t>(m_token_bitmask_ov.get_size())};
    m_token_bitmask->shape = &m_bitmask_shape[0];
    
    m_next_token_logits = std::make_shared<DLTensor>();
    m_next_token_logits->device = DLDevice{kDLCPU, 0};
    m_next_token_logits->ndim = 1;
    m_next_token_logits->dtype = DLDataType{kDLFloat, 32, 1};
    m_next_token_logits->byte_offset = 0;
    m_logits_shape = {static_cast<int64_t>(m_vocab_size)};
    m_next_token_logits->shape = &m_logits_shape[0];
    m_next_token_logits->strides = &m_logits_strides[0];
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
