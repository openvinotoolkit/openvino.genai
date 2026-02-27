// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <filesystem>
#include <fstream>
#include <memory>

#include "minja/minja.hpp"
#include "minja/chat-template.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

#include "gguf_utils/gguf_tokenizer.hpp"
#include "tokenizer/chat_template_fallback_map.hpp"
#include "tokenizer/make_tokenizer_stateful.hpp"
#include "tokenizer/tokenizers_path.hpp"
#include "circular_buffer_queue.hpp"
#include "json_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
void check_arguments(const ov::AnyMap& parameters, std::set<std::string> allowed_argnames);
ov::Core core_with_extension();
ov::Core get_core_singleton();

class StructuredOutputController;
class Tokenizer::TokenizerImpl {
public:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_tokenizer;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_detokenizer;
    std::unordered_map<ov::InferRequest*, ov::AnyMap> m_request_to_state_flags;
    std::shared_ptr<void> m_shared_object_ov_tokenizers = nullptr;
    bool is_paired_input = false;
    bool m_older_than_24_5 = false;
    int64_t m_pad_token_id = -1;
    int64_t m_bos_token_id = -1;
    int64_t m_eos_token_id = -1;
    std::string m_pad_token = {};
    std::string m_bos_token = {};
    std::string m_eos_token = {};
    std::string m_chat_template = {};
    std::string m_original_chat_template = {};
    std::vector<std::string> m_vocab = {};
    std::shared_ptr<StructuredOutputController> m_structured_output_controller = nullptr;

    template <typename T>
    void set_state_value(ov::VariableState& state, std::optional<T> value, ov::AnyMap& state_flags);

    void set_state_if_necessary(CircularBufferQueueElementGuard<ov::InferRequest>& infer_request_guard, const ov::AnyMap& params);

    TokenizerImpl(const std::filesystem::path& models_path, const ov::AnyMap& properties);
    TokenizerImpl(const std::pair<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>>& models, const ov::AnyMap& properties);

    void setup_tokenizer(const std::filesystem::path& models_path, const ov::AnyMap& properties);
    void setup_tokenizer(const std::pair<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>>& models, ov::AnyMap properties);

    void read_config(const std::filesystem::path& tokenizer_path);
    void read_special_tokens_map(const std::filesystem::path& tokenizer_path);
    void read_tokenizer_config_if_necessary(const std::filesystem::path& tokenizer_path);
    void infer_special_tokens_if_necessary();

    TokenizedInputs encode(const std::string& prompt, const ov::AnyMap& tokenization_params = {});
    TokenizedInputs encode(const std::vector<std::pair<std::string, std::string>>& prompts_pairs, const ov::AnyMap& tokenization_params = {});
    TokenizedInputs encode(const std::vector<std::string>& prompts_1, const std::vector<std::string>& prompts_2, const ov::AnyMap& tokenization_params = {});
    TokenizedInputs encode(const std::vector<std::string>& prompts, const ov::AnyMap& tokenization_params = {});

    TokenizedInputs get_copied_results(ov::Tensor input_ids, ov::Tensor attention_mask);

    std::string decode(const std::vector<int64_t>& tokens, const ov::AnyMap& detokenization_params = {});
    std::vector<std::string> decode(const ov::Tensor& tokens, const ov::AnyMap& detokenization_params = {});
    std::vector<std::string> decode(const std::vector<std::vector<int64_t>>& lines, const ov::AnyMap& detokenization_params = {});

    std::string apply_chat_template(const ChatHistory& history,
                                    bool add_generation_prompt,
                                    const std::string& chat_template,
                                    const std::optional<JsonContainer>& tools,
                                    const std::optional<JsonContainer>& extra_context) const;

    void set_chat_template(const std::string& chat_template);
    std::string get_chat_template() const;
    std::string get_original_chat_template() const;
    std::shared_ptr<StructuredOutputController> get_structured_output_controller(std::optional<int> vocab_size = std::nullopt);
};

}  // namespace genai
}  // namespace ov
