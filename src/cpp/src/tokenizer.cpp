// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <memory>
#include <jinja2cpp/template.h>
#include <jinja2cpp/template_env.h>
#include <jinja2cpp/user_callable.h>
#include <jinja2cpp/generic_list.h>
#include <jinja2cpp/generic_list_iterator.h>

#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/genai/tokenizer.hpp"

#include "make_combine_segments_stateful.hpp"
#include "tokenizers_path.hpp"
#include "circular_buffer_queue.hpp"
#include "json_utils.hpp"
#include "utils.hpp"

namespace {

// todo: remove when openvino-tokenizers will support left padding
ov::genai::TokenizedInputs pad_left(ov::Tensor& input_ids, ov::Tensor& attention_mask) {
    const size_t batch_size = input_ids.get_shape()[0];
    const size_t sequence_length = input_ids.get_shape()[1];
    int64_t* inputs_data = input_ids.data<int64_t>();
    int64_t* attention_mask_data = attention_mask.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;

        // last token in the sequence is not a PAD_TOKEN, skipping
        if (attention_mask_data[batch_offset + sequence_length - 1] == 1)
            continue;

        size_t pad_tokens_number = 0;
        for (int i = sequence_length - 1; i >= 0; i--) {
            const size_t token_offset = batch_offset + i;

            // count pad tokens
            if (attention_mask_data[token_offset] == 0)
                continue;

            if (pad_tokens_number == 0)
                pad_tokens_number = sequence_length - i - 1;

            std::swap(inputs_data[token_offset], inputs_data[token_offset + pad_tokens_number]);
            std::swap(attention_mask_data[token_offset], attention_mask_data[token_offset + pad_tokens_number]);
        }
    }

    return {input_ids, attention_mask};
}

}  // namespace

namespace ov {
namespace genai {

class Tokenizer::TokenizerImpl {
public:
    ov::CompiledModel m_tokenizer;
    ov::CompiledModel m_detokenizer;
    std::shared_ptr<ov::Core> m_core = nullptr;
    
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_tokenizer;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_detokenizer;
    // To change the adding special tokens mode we use a statefull subgraph, 
    // this flag holds the current state value of the CompiledModel.
    bool m_add_special_tokens = true;  
    bool m_older_than_24_5 = false;
    
    int64_t m_pad_token_id = -1;
    int64_t m_bos_token_id = -1;
    int64_t m_eos_token_id = -1;

    std::string m_pad_token = {};
    std::string m_bos_token = {};
    std::string m_eos_token = {};

    std::string m_chat_template = {};

    void set_state_if_necessary(CircularBufferQueueElementGuard<ov::InferRequest>& infer_request_guard, bool add_special_tokens) {
        // If user requested add_special_tokens mode different from the current one,
        // need to set state variable.
        // If requested mode matches the stored state set, then don't touch states.
        if (add_special_tokens == m_add_special_tokens) {
            return;
        }
        if (m_older_than_24_5) {
            // Changing add_special_tokens at runtime was introduced in
            // 24.5. Older tokenizers still allow manipulating their
            // state but the effect is incorrect.
            return;
        }
        
        // auto states = m_ireq_queue_tokenizer->get(0).query_state();
        ov::Tensor add_special_tensor = ov::Tensor(ov::element::boolean, {});
        *add_special_tensor.data<bool>() = add_special_tokens;

        for (auto& state: infer_request_guard.get().query_state()) {
            if (state.get_name().find(ov::genai::ADD_SPECIAL_TOKENS_VAR_ID) == std::string::npos) {
                // It's not add_special_tokens flag state.
                continue;
            }
            state.set_state(add_special_tensor);
            break;            
        }
        m_add_special_tokens = add_special_tokens;
    }

    TokenizerImpl() = default;

    std::shared_ptr<ov::Core> get_core() {
        if (m_core) {
            return m_core;
        }
        m_core = std::make_shared<ov::Core>();
        const char* ov_tokenizer_path = getenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME);
        OPENVINO_ASSERT(ov_tokenizer_path, "openvino_tokenizers path is not set");
        m_core->add_extension(ov_tokenizer_path);
        return m_core;
    }

    TokenizerImpl(std::filesystem::path tokenizer_path, const ov::AnyMap& properties) {
        // TODO: get chat template
        // TODO: get special tokens

        OPENVINO_ASSERT(tokenizer_path.extension() != ".xml", "'tokenizer_path' parameter should be a path to a dir not a xml file");

        std::shared_ptr<ov::Model> ov_tokenizer = nullptr;
        std::shared_ptr<ov::Model> ov_detokenizer = nullptr;

        if (std::filesystem::exists(tokenizer_path / "openvino_tokenizer.xml")) {
            ov_tokenizer = get_core()->read_model(tokenizer_path / "openvino_tokenizer.xml");
        }
        
        if (std::filesystem::exists(tokenizer_path / "openvino_detokenizer.xml")) {
            ov_detokenizer = get_core()->read_model(tokenizer_path / "openvino_detokenizer.xml");
        }
        // TODO: reimplement without this hack.
        *this = TokenizerImpl(std::make_pair(ov_tokenizer, ov_detokenizer), properties);
    }

    TokenizerImpl(const std::pair<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>>& models,  const ov::AnyMap& properties) {
        auto [ov_tokenizer, ov_detokenizer] = models;

        m_older_than_24_5 = ov_tokenizer->get_rt_info().count("openvino_tokenizers_version") != 1;
        auto core = *get_core();
        std::string device = "CPU"; // only CPU is supported for now
        if (ov_tokenizer) {
            ov::pass::Manager manager;
            manager.register_pass<MakeCombineSegmentsSatateful>();
            manager.run_passes(ov_tokenizer);
            m_tokenizer = core.compile_model(ov_tokenizer, device, properties);

            m_ireq_queue_tokenizer = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
                m_tokenizer.get_property(ov::optimal_number_of_infer_requests),
                [this]() -> ov::InferRequest {
                    return std::move(this->m_tokenizer.create_infer_request());
                });
        }

        if (ov_detokenizer) {
            m_detokenizer = core.compile_model(ov_detokenizer, device, properties);

            m_ireq_queue_detokenizer = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
                m_detokenizer.get_property(ov::optimal_number_of_infer_requests),
                [this]() -> ov::InferRequest {
                    return std::move(this->m_detokenizer.create_infer_request());
                });
        }
        
        // Initialize tokenizer's cache to save time later.
        // TODO CVS-150630: Empty strings sporadically can fail, therefore use nonempty string for warmup.
        auto tokenized_input = encode("non empty string").input_ids;
        if (m_detokenizer)
            decode(tokenized_input);

        utils::read_rt_info(ov_tokenizer, "chat_template", m_chat_template);
        utils::read_rt_info(ov_tokenizer, "pad_token_id", m_pad_token_id);
        utils::read_rt_info(ov_tokenizer, "bos_token_id", m_bos_token_id);
        utils::read_rt_info(ov_tokenizer, "eos_token_id", m_eos_token_id);

        m_pad_token = decode(std::vector{m_pad_token_id});
        m_bos_token = decode(std::vector{m_bos_token_id});
        m_eos_token = decode(std::vector{m_eos_token_id});
    }

    TokenizerImpl(
        std::string& tokenizer_model_str,
        ov::Tensor& tokenizer_weights_tensor,
        std::string& detokenizer_model_str,
        ov::Tensor& detokenizer_weights_tensor,
        const ov::AnyMap& properties
    ) {
        auto core = *get_core();

        auto ov_tokenizer = core.read_model(tokenizer_model_str, tokenizer_weights_tensor);
        auto ov_detokenize = core.read_model(detokenizer_model_str, detokenizer_weights_tensor);
        // TODO: reimplement without this hack.
        *this = TokenizerImpl(std::make_pair(ov_tokenizer, ov_detokenize), properties);
    }

    TokenizerImpl(std::string& model_str, ov::Tensor& weights_tensor, const ov::AnyMap& properties = {}) {
        auto core = *get_core();
        auto model = core.read_model(model_str, weights_tensor);
        
        auto parameters = model->get_parameters();
        OPENVINO_ASSERT(!parameters.empty());
        if (parameters.front()->get_element_type() == ov::element::string) {
            // It's a tokenizer
            // TODO: reimplement without this hack.
            *this = TokenizerImpl(std::make_pair(model, nullptr), properties);
        } else {
            // It's a detokenizer
            // TODO: reimplement without this hack.
            *this = TokenizerImpl(std::make_pair(nullptr, model), properties);
        }
    }

    TokenizedInputs encode(std::string prompt, const ov::AnyMap& tokenization_params = {}) {
        bool add_special_tokens_flag = true;
        ov::genai::utils::read_anymap_param(tokenization_params, add_special_tokens.name(), add_special_tokens_flag);

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_tokenizer.get());
        set_state_if_necessary(infer_request_guard, add_special_tokens_flag);
        size_t batch_size = 1;
        infer_request_guard.get().set_input_tensor(ov::Tensor{ov::element::string, {batch_size}, &prompt});
        infer_request_guard.get().start_async();
        infer_request_guard.get().wait();
        return get_copied_results(
            infer_request_guard.get().get_tensor("input_ids"),
            infer_request_guard.get().get_tensor("attention_mask")
        );
    }

    TokenizedInputs encode(std::vector<std::string>& prompts, const ov::AnyMap& tokenization_params = {}) {
        TokenizedInputs unpadded;
        {
            bool add_special_tokens_flag = true;
            ov::genai::utils::read_anymap_param(tokenization_params, add_special_tokens.name(), add_special_tokens_flag);

            CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_tokenizer.get());
            set_state_if_necessary(infer_request_guard, add_special_tokens_flag);
            infer_request_guard.get().set_input_tensor(ov::Tensor{ov::element::string, {prompts.size()}, prompts.data()});
            auto size_ = infer_request_guard.get().get_input_tensor().get_shape();
            infer_request_guard.get().start_async();
            infer_request_guard.get().wait();

            unpadded = get_copied_results(
                infer_request_guard.get().get_tensor("input_ids"),
                infer_request_guard.get().get_tensor("attention_mask")
            );
        }
        return pad_left(unpadded.input_ids, unpadded.attention_mask);
    }

    TokenizedInputs get_copied_results(ov::Tensor input_ids, ov::Tensor attention_mask) {
        ov::Tensor input_ids_ = ov::Tensor(input_ids.get_element_type(), input_ids.get_shape());
        ov::Tensor attention_mask_ = ov::Tensor(attention_mask.get_element_type(), attention_mask.get_shape());
        input_ids.copy_to(input_ids_);
        attention_mask.copy_to(attention_mask_);

        return {input_ids_, attention_mask_};
    }

    std::string decode(std::vector<int64_t> tokens) {
        OPENVINO_ASSERT(m_detokenizer, "Detokenize model has not been provided. Tokenizer::decode is not available");

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_detokenizer.get());
        size_t batch_size = 1;
        infer_request_guard.get().set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
        infer_request_guard.get().start_async();
        infer_request_guard.get().wait();
        return infer_request_guard.get().get_output_tensor().data<std::string>()[0];
    }

    std::vector<std::string> decode(ov::Tensor tokens) {
        OPENVINO_ASSERT(m_detokenizer, "Detokenize model has not been provided. Tokenizer::decode is not available");
        OPENVINO_ASSERT(tokens.get_element_type() == ov::element::i64, "tokens tensor element type should be an i64");
        OPENVINO_ASSERT(tokens.get_shape().size() == 2, "tokens tensor should of rank 2 with shape [batch_size, seq_len]");

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_detokenizer.get());
        infer_request_guard.get().set_input_tensor(tokens);
        infer_request_guard.get().start_async();
        infer_request_guard.get().wait();

        auto res = infer_request_guard.get().get_output_tensor();
        auto res_data = res.data<std::string>();
        return std::vector<std::string>(res_data, res_data + res.get_shape()[0]);
    }

    std::vector<std::string> decode(std::vector<std::vector<int64_t>> lines) {
        OPENVINO_ASSERT(m_detokenizer, "Detokenize model has not been provided. Tokenizer::decode is not available");

        auto compare_lengths = [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
            return a.size() < b.size();
        };
        size_t max_len = std::max_element(lines.begin(), lines.end(), compare_lengths)->size();

        ov::Tensor tokens = ov::Tensor{ov::element::i64, {lines.size(), max_len}};
        auto tokens_data = tokens.data<int64_t>();

        for (size_t i = 0; i < lines.size(); ++i) {
            const auto& line = lines[i];
            size_t line_len = line.size();
            std::copy(line.begin(), line.end(), tokens_data + i * max_len);
            std::fill(tokens_data + i * max_len + line_len, tokens_data + (i + 1) * max_len, m_pad_token_id);
        }

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_detokenizer.get());
        infer_request_guard.get().set_input_tensor(tokens);
        infer_request_guard.get().start_async();
        infer_request_guard.get().wait();
        auto res = infer_request_guard.get().get_output_tensor();
        auto res_data = res.data<std::string>();
        return std::vector<std::string>(res_data, res_data + res.get_shape()[0]);
    }

    std::string patch_chat_template(std::string template_str) const {
        // Replace what jinja2cpp doesn't support
        std::pair<std::string, std::string> replace_str_map[] = {
            {"'}", "' }"},
            {"{'", "{ '"},
            {".strip()", ""},
            {"is not none", "is defined"},
            {"is none", "is undefined"},
            {"= none", "= undefined"},
            // Jinja2Cpp does not support Python-style slicing, e.g. [1:].
            // If chat template contains such slicing, we replace it with 
            // a placeholder at the moment.
            {"messages[1:]", "slice(messages, 1)"},
        };

        for (const auto& [from, to] : replace_str_map) {
            size_t pos = 0;
            while ((pos = template_str.find(from, pos)) != std::string::npos) {
                template_str.replace(pos, from.size(), to);
                pos += to.size();
            }
        }
        return template_str;
    }

    std::string apply_chat_template(ChatHistory history,
                                    bool add_generation_prompt,
                                    const std::string& chat_template) const {
        std::string chat_tpl = chat_template.empty() ? m_chat_template : patch_chat_template(chat_template);
        OPENVINO_ASSERT(!chat_tpl.empty(),
                        "Chat template wasn't found. This may indicate that the model wasn't trained for chat scenario."
                        " Please add 'chat_template' to tokenizer_config.json to use the model in chat scenario."
                        " For more information see the section Troubleshooting in README.md");
        jinja2::TemplateEnv env;
        env.GetSettings().lstripBlocks = true;
        env.GetSettings().trimBlocks = true;
        jinja2::Template tpl(&env);
        tpl.Load(chat_tpl);
        
        jinja2::UserCallable slice_callable = jinja2::MakeCallable(
            [](const jinja2::GenericList& messages, const size_t& start) {
                jinja2::ValuesList result;

                size_t iter_num = 0;
                for (auto message = messages.begin(); message != messages.end(); message++, iter_num++) {
                    if (iter_num < start)
                        continue;
                    result.emplace_back(*message);
                }

                return result;
            },
            jinja2::ArgInfo{"messages"}, jinja2::ArgInfo{"start"}
        );

        jinja2::ValuesList jinja_messages;
        jinja2::ValuesMap jinja_message;
        for (const auto& message : history) {
            jinja_message = {{"role", message.at("role")}, {"content", message.at("content")}};
            jinja_messages.emplace_back(jinja_message);
        }

        jinja2::ValuesMap params = {
            {"messages", jinja_messages},
            {"bos_token",  m_bos_token},
            {"eos_token", m_eos_token},
            {"pad_token", m_pad_token},
            {"add_generation_prompt", add_generation_prompt},
            {"slice", slice_callable},
        };

        try {
            return tpl.RenderAsString(params).value();
        } catch (const std::exception& error) {
            OPENVINO_THROW("Chat template for the current model is not supported by Jinja2Cpp. "
                           "Please apply template manually to your prompt before calling generate. "
                           "For example: <start_of_turn>user{user_prompt}<end_of_turn><start_of_turn>model");
        }
    }

    void set_chat_template(const std::string& chat_template) {
        m_chat_template = patch_chat_template(chat_template);
    }
};

Tokenizer::Tokenizer(const std::filesystem::path& tokenizer_path, const ov::AnyMap& properties) {
    ScopedVar env_manager(tokenizers_relative_to_genai().string());
    m_pimpl = std::make_shared<TokenizerImpl>(tokenizer_path, properties);
}

Tokenizer::Tokenizer(
    std::string& tokenizer_model_str,
    ov::Tensor& tokenizer_weights_tensor,
    std::string& detokenizer_model_str,
    ov::Tensor&  detokenizer_weights_tensor,
    const ov::AnyMap& properties
) {
    m_pimpl = std::make_shared<TokenizerImpl>(
        tokenizer_model_str,
        tokenizer_weights_tensor,
        detokenizer_model_str,
        detokenizer_weights_tensor,
        properties
    );
}

Tokenizer::Tokenizer(std::string& model_str, ov::Tensor& weights_tensor, const ov::AnyMap& properties) {
    m_pimpl = std::make_shared<TokenizerImpl>(model_str, weights_tensor, properties);
}

TokenizedInputs Tokenizer::encode(const std::string prompt, const ov::AnyMap& tokenization_params) {
    return m_pimpl->encode(std::move(prompt), tokenization_params);
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>& prompts, const ov::AnyMap& tokenization_params) {
    return m_pimpl->encode(prompts, tokenization_params);
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>&& prompts, const ov::AnyMap& tokenization_params) {
    return m_pimpl->encode(prompts, tokenization_params);
}

TokenizedInputs Tokenizer::encode(std::initializer_list<std::string>& text, const ov::AnyMap& tokenization_params) {
    return encode(std::vector<std::string>(text.begin(), text.end()), tokenization_params);
}

std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(ov::Tensor tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(std::vector<std::vector<int64_t>> lines) {
    return m_pimpl->decode(lines);
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

std::string Tokenizer::apply_chat_template(ChatHistory history,
                                           bool add_generation_prompt,
                                           const std::string& chat_template) const {
    return m_pimpl->apply_chat_template(history, add_generation_prompt, chat_template);
}

void Tokenizer::set_chat_template(const std::string& chat_template) {
    m_pimpl->set_chat_template(chat_template);
}

Tokenizer::~Tokenizer() = default;
}  // namespace genai
}  // namespace ov
