// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <memory>

#include "minja/minja.hpp"
#include "minja/chat-template.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/genai/tokenizer.hpp"

#include "tokenizer/tokenizer_impl.hpp"

namespace ov {
namespace genai {

Tokenizer::Tokenizer(const std::filesystem::path& tokenizer_path, const ov::AnyMap& properties) {
    m_pimpl = std::make_shared<Tokenizer::TokenizerImpl>(tokenizer_path, properties);
}

Tokenizer::Tokenizer(
    const std::string& tokenizer_model_str,
    const ov::Tensor& tokenizer_weights_tensor,
    const std::string& detokenizer_model_str,
    const ov::Tensor&  detokenizer_weights_tensor,
    const ov::AnyMap& properties
) {
    ScopedVar env_manager(tokenizers_relative_to_genai());
    auto core = get_core_singleton();

    auto ov_tokenizer = core.read_model(tokenizer_model_str, tokenizer_weights_tensor);
    auto ov_detokenizer = core.read_model(detokenizer_model_str, detokenizer_weights_tensor);
    m_pimpl = std::make_shared<Tokenizer::TokenizerImpl>(std::make_pair(ov_tokenizer, ov_detokenizer), properties);
}

Tokenizer::Tokenizer(const std::string& model_str, ov::Tensor& weights_tensor, const ov::AnyMap& properties) {
    ScopedVar env_manager(tokenizers_relative_to_genai());
    auto core = get_core_singleton();
    auto model = core.read_model(model_str, weights_tensor);

    auto parameters = model->get_parameters();
    OPENVINO_ASSERT(!parameters.empty());
    if (parameters.front()->get_element_type() == ov::element::string) {
        // It's a tokenizer
        m_pimpl = std::make_shared<Tokenizer::TokenizerImpl>(std::make_pair(model, nullptr), properties);
    } else {
        // It's a detokenizer
        m_pimpl = std::make_shared<Tokenizer::TokenizerImpl>(std::make_pair(nullptr, model), properties);
    }
}

TokenizedInputs Tokenizer::encode(const std::string& prompt, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name(),
                                          ov::genai::max_length.name(),
                                          ov::genai::pad_to_max_length.name(),
                                          ov::genai::padding_side.name()});
    return m_pimpl->encode(prompt, tokenization_params);
}

TokenizedInputs Tokenizer::encode(const std::vector<std::pair<std::string, std::string>>& prompts, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name(),
                                          ov::genai::max_length.name(),
                                          ov::genai::pad_to_max_length.name(),
                                          ov::genai::padding_side.name()});
    return m_pimpl->encode(prompts, tokenization_params);
}

TokenizedInputs Tokenizer::encode(const std::vector<std::string>& prompts_1, const std::vector<std::string>& prompts_2, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name(),
                                          ov::genai::max_length.name(),
                                          ov::genai::pad_to_max_length.name(),
                                          ov::genai::padding_side.name()});
    return m_pimpl->encode(prompts_1, prompts_2, tokenization_params);
}

TokenizedInputs Tokenizer::encode(const std::vector<std::string>& prompts, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name(),
                                          ov::genai::max_length.name(),
                                          ov::genai::pad_to_max_length.name(),
                                          ov::genai::padding_side.name()});
    return m_pimpl->encode(prompts, tokenization_params);
}

TokenizedInputs Tokenizer::encode(const std::initializer_list<std::string>& text, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name(),
                                          ov::genai::max_length.name(),
                                          ov::genai::pad_to_max_length.name(),
                                          ov::genai::padding_side.name()});
    return encode(std::vector<std::string>(text.begin(), text.end()), tokenization_params);
}

std::string Tokenizer::decode(const std::vector<int64_t>& tokens, const ov::AnyMap& detokenization_params) {
    check_arguments(detokenization_params, {ov::genai::skip_special_tokens.name()});
    return m_pimpl->decode(tokens, detokenization_params);
}

std::vector<std::string> Tokenizer::decode(const ov::Tensor& tokens, const ov::AnyMap& detokenization_params) {
    check_arguments(detokenization_params, {ov::genai::skip_special_tokens.name()});
    return m_pimpl->decode(tokens, detokenization_params);
}

std::vector<std::string> Tokenizer::decode(const std::vector<std::vector<int64_t>>& lines, const ov::AnyMap& detokenization_params) {
    check_arguments(detokenization_params, {ov::genai::skip_special_tokens.name()});
    return m_pimpl->decode(lines, detokenization_params);
}

int64_t Tokenizer::get_bos_token_id() const {
    return m_pimpl->m_bos_token_id;
}

int64_t Tokenizer::get_eos_token_id() const {
    return m_pimpl->m_eos_token_id;
}

int64_t Tokenizer::get_pad_token_id() const {
    return m_pimpl->m_pad_token_id;
}

std::string Tokenizer::get_pad_token() const {
    return m_pimpl->m_pad_token;
}

std::string Tokenizer::get_bos_token() const {
    return m_pimpl->m_bos_token;
}

std::string Tokenizer::get_eos_token() const {
    return m_pimpl->m_eos_token;
}

std::string Tokenizer::apply_chat_template(const ChatHistory& history,
                                           bool add_generation_prompt,
                                           const std::string& chat_template,
                                           const std::optional<JsonContainer>& tools,
                                           const std::optional<JsonContainer>& extra_context) const {
    return m_pimpl->apply_chat_template(history, add_generation_prompt, chat_template, tools, extra_context);
}

std::string Tokenizer::get_chat_template() const {
    return m_pimpl->get_chat_template();
}

std::string Tokenizer::get_original_chat_template() const {
    return m_pimpl->get_original_chat_template();
}

void Tokenizer::set_chat_template(const std::string& chat_template) {
    m_pimpl->set_chat_template(chat_template);
}

Vocab Tokenizer::get_vocab() const {
    const auto& vocab_vector = get_vocab_vector();

    Vocab vocab;
    vocab.reserve(vocab_vector.size());
    for (size_t i = 0; i < vocab_vector.size(); ++i) {
        vocab[vocab_vector[i]] = i;
    }
    return vocab;
}

const std::vector<std::string>& Tokenizer::get_vocab_vector() const {
    OPENVINO_ASSERT(m_pimpl != nullptr, "Tokenizer is not initialized. Please check if the tokenizer model was provided and loaded correctly.");
    OPENVINO_ASSERT(!m_pimpl->m_vocab.empty(), "Tokenizer vocab is empty. Please check if the detokenizer model was provided and loaded correctly.");
    return m_pimpl->m_vocab;
}

bool Tokenizer::supports_paired_input() const {
    return m_pimpl->is_paired_input;
}

Tokenizer::~Tokenizer() {
    m_pimpl.reset();

    // release CPU plugin ()
    try {
        get_core_singleton().unload_plugin("CPU");
    } catch (const ov::Exception&) {
        // Note: in a theory it can throw an exception when 2 different Tokenizers are created from
        // different threads and then both of them unload plugin for 'device' from ov::Core
    }
}

}  // namespace genai
}  // namespace ov
