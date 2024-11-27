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

#include "make_tokenizer_stateful.hpp"
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

void check_arguments(const ov::AnyMap& parameters, std::set<std::string> allowed_argnames) {
    for (const auto& [key, value] : parameters) {
        if (allowed_argnames.find(key) == allowed_argnames.end()) {
            OPENVINO_THROW("unacceptable parameter key: " + key);
        }
    }
}

constexpr char bos_token_key_name[] = "bos_token";
constexpr char eos_token_key_name[] = "eos_token";
constexpr char pad_token_key_name[] = "pad_token";

}  // namespace

namespace ov {
namespace genai {

class Tokenizer::TokenizerImpl {
public:
    ov::CompiledModel m_tokenizer;
    ov::CompiledModel m_detokenizer;

    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_tokenizer;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_detokenizer;
    // To change the adding special tokens mode we use a statefull subgraph, 
    // this flag holds the current state value of the CompiledModel.
    bool m_add_special_tokens = true;
    bool m_skip_special_tokens = true;
    bool m_older_than_24_5 = false;
    
    int64_t m_pad_token_id = -1;
    int64_t m_bos_token_id = -1;
    int64_t m_eos_token_id = -1;

    std::string m_pad_token = {};
    std::string m_bos_token = {};
    std::string m_eos_token = {};

    std::string m_chat_template = {};

    void set_state_if_necessary(CircularBufferQueueElementGuard<ov::InferRequest>& infer_request_guard, const ov::AnyMap& params) {
        bool add_special_tokens_flag = m_add_special_tokens;
        bool skip_special_tokens_flag = m_skip_special_tokens;
        ov::genai::utils::read_anymap_param(params, add_special_tokens.name(), add_special_tokens_flag);
        ov::genai::utils::read_anymap_param(params, skip_special_tokens.name(), skip_special_tokens_flag);

        // If user requested add_special_tokens mode different from the current one,
        // need to set state variable.
        // If requested mode matches the stored state set, then don't touch states.
        if (add_special_tokens_flag == m_add_special_tokens && skip_special_tokens_flag == m_skip_special_tokens) {
            return;
        }
        if (m_older_than_24_5) {
            // Changing add_special_tokens at runtime was introduced in
            // 24.5. Older tokenizers still allow manipulating their
            // state but the effect is incorrect.
            return;
        }
        
        // add_special_tokens is managed by Select op with a bool input.
        ov::Tensor add_special_tensor = ov::Tensor(ov::element::boolean, {});
        *add_special_tensor.data<bool>() = add_special_tokens_flag;
        
        // skip_special_tokens is managed by multiplication with a number, therefore i32.
        ov::Tensor skip_special_tensor = ov::Tensor(ov::element::i32, {1});
        *skip_special_tensor.data<int>() = skip_special_tokens_flag;

        for (auto& state: infer_request_guard.get().query_state()) {
            if (state.get_name().find(ov::genai::ADD_SPECIAL_TOKENS_VAR_ID) != std::string::npos) {
                state.set_state(add_special_tensor);
            } else if (state.get_name().find(ov::genai::SKIP_SPECIAL_TOKENS_VAR_ID) != std::string::npos) {
                state.set_state(skip_special_tensor);
            }
        }
        m_add_special_tokens = add_special_tokens_flag;
        m_skip_special_tokens = skip_special_tokens_flag;
    }

    TokenizerImpl() = default;

    TokenizerImpl(std::filesystem::path tokenizer_path, const ov::AnyMap& properties)
        : m_chat_template{chat_template_from_tokenizer_json_if_exists(tokenizer_path)} {
        ov::Core core;

        OPENVINO_ASSERT(tokenizer_path.extension() != ".xml", "'tokenizer_path' parameter should be a path to a dir not a xml file");

        const char* ov_tokenizer_path = getenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME);
        OPENVINO_ASSERT(ov_tokenizer_path, "openvino_tokenizers path is not set");
        core.add_extension(ov_tokenizer_path);

        read_config(tokenizer_path);
        read_special_tokens_map(tokenizer_path);

        // Try to read tokenizer_config if some token ids or token str are not defined.
        read_tokenizer_config_if_necessary(tokenizer_path);

        auto device = "CPU"; // currently openvino_tokenizer supports only CPU
        auto ov_tokenizer = core.read_model(tokenizer_path / "openvino_tokenizer.xml");
        std::shared_ptr<ov::Model> ov_detokenizer = nullptr;
        if (std::filesystem::exists(tokenizer_path / "openvino_detokenizer.xml")) {
            ov_detokenizer = core.read_model(tokenizer_path / "openvino_detokenizer.xml");
        }
        m_older_than_24_5 = ov_tokenizer->get_rt_info().count("openvino_tokenizers_version") != 1;
        
        ov::pass::Manager manager_tok;
        manager_tok.register_pass<MakeCombineSegmentsSatateful>();
        manager_tok.run_passes(ov_tokenizer);
        
        m_tokenizer = core.compile_model(ov_tokenizer, device, properties);
        if (ov_detokenizer) {
            ov::pass::Manager manager_detok;
            manager_detok.register_pass<MakeVocabDecoderSatateful>();
            manager_detok.run_passes(ov_detokenizer);
            m_detokenizer = core.compile_model(ov_detokenizer, device, properties);
        }

        const size_t INFER_REQUEST_QUEUE_SIZE = m_tokenizer.get_property(ov::optimal_number_of_infer_requests);
        m_ireq_queue_tokenizer = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            INFER_REQUEST_QUEUE_SIZE,
            [this]() -> ov::InferRequest {
                return std::move(this->m_tokenizer.create_infer_request());
            });
        if (m_detokenizer) {
            m_ireq_queue_detokenizer = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
                INFER_REQUEST_QUEUE_SIZE,
                [this]() -> ov::InferRequest {
                    return std::move(this->m_detokenizer.create_infer_request());
                });
        }

        // Get special token ids by inference if they are not defined.
        infer_special_tokens_if_necessary();
        // Initialize tokenizer's cache to save time later.
        // infer_special_tokens_if_necessary() already could do that
        // but it didn't run decode() for sure.
        // TODO CVS-150630: Empty strings sporadically can fail, therefore use nonempty string for warmup.
        auto tokenized_input = encode("non empty string").input_ids;
        if (m_detokenizer)
            decode(tokenized_input);
    }

    // load special tokens ids from config.json
    void read_config(const std::filesystem::path& tokenizer_path) {
        auto config_file_path = tokenizer_path / "config.json";
        if (!std::filesystem::exists(config_file_path))
            return ;
        std::ifstream file(config_file_path);
        if (!file.is_open())
            return ;

        nlohmann::json data = nlohmann::json::parse(file);
        using ov::genai::utils::read_json_param;

        read_json_param(data, "pad_token_id", m_pad_token_id);
        read_json_param(data, "bos_token_id", m_bos_token_id);
        read_json_param(data, "eos_token_id", m_eos_token_id);
    }

    // Reads the string representation of special tokens if they exist.
    void read_special_tokens_map(const std::filesystem::path& tokenizer_path) {
        auto special_tokens_file_path = tokenizer_path / "special_tokens_map.json";
        if (!std::filesystem::exists(special_tokens_file_path))
            return ;
        std::ifstream f(special_tokens_file_path);
        if (!f.is_open())
            return ;

        nlohmann::json data = nlohmann::json::parse(f);

        using ov::genai::utils::read_json_param;
        // they are in the format {"bos_token": { "content": "<s>",... }}
        auto read_token_content_str = [&data](std::string key_name, std::string& val) {
            if (val == "" && data.contains(key_name)) { read_json_param(data[key_name], "content", val); }
        };
        read_token_content_str(pad_token_key_name, m_pad_token);
        read_token_content_str(bos_token_key_name, m_bos_token);
        read_token_content_str(eos_token_key_name, m_eos_token);
    }

    // Read string representation of special tokens if they exist.
    // Also tries to load special token ids from added_tokens_decoder if they exist.
    // Will not override special token strings or ids if they already exist.
    void read_tokenizer_config_if_necessary(const std::filesystem::path& tokenizer_path) {
        if (m_pad_token_id != -1 && m_bos_token_id != -1 && m_eos_token_id != -1 &&
            !m_pad_token.empty() && !m_bos_token.empty() && !m_eos_token.empty()) {
            return ;
        }

        auto tokenizer_config_file_path = tokenizer_path / "tokenizer_config.json";
        if (!std::filesystem::exists(tokenizer_config_file_path))
            return ;
        std::ifstream f(tokenizer_config_file_path);
        if (!f.is_open())
            return ;

        nlohmann::json data = nlohmann::json::parse(f);

        // read special tokens string representation
        // if they are presented directly {"bos_token": "<bos>"}
        using ov::genai::utils::read_json_param;
        auto read_token_str = [&data](std::string key_name, std::string& val) {
            if (val.empty()) { read_json_param(data, key_name, val); }
        };
        read_token_str(pad_token_key_name, m_pad_token);
        read_token_str(bos_token_key_name, m_bos_token);
        read_token_str(eos_token_key_name, m_eos_token);

        // if special tokens are not loaded directly, try to read
        // if they are in the format {"bos_token": { "content": "<s>",... }}
        auto read_token_content_str = [&data](std::string key_name, std::string& val) {
            if (val.empty() && data.contains(key_name)) { read_json_param(data[key_name], "content", val); }
        };
        read_token_content_str(pad_token_key_name, m_pad_token);
        read_token_content_str(bos_token_key_name, m_bos_token);
        read_token_content_str(eos_token_key_name, m_eos_token);

        // if pad_token not found use eos_token as pad_token
        if (m_pad_token.empty() && !m_eos_token.empty()) {
            m_pad_token = m_eos_token;
        }

        // special token ids integer representation are already defined
        if (m_pad_token_id != -1 && m_bos_token_id != -1 && m_eos_token_id != -1)
            return ;

        // values are stored as {"added_tokens_decoder": {"0": {"content": "<pad>"}}}
        // token id is a key in the form of a string, need to do std::stoi
        std::string spec_tokens_key_name = "added_tokens_decoder";
        if (!data.contains(spec_tokens_key_name))
            return ;

        // if added_tokens_decoder has different format items() will not fail
        for (auto& [key, value] : data[spec_tokens_key_name].items()) {
            if (!value.contains("content"))
                continue;
            auto content = value["content"];
            if (m_pad_token_id == -1 && content == m_pad_token)
                m_pad_token_id = std::stoi(key);
            if (m_bos_token_id == -1 && content == m_bos_token)
                m_bos_token_id = std::stoi(key);
            if (m_eos_token_id == -1 && content == m_eos_token)
                m_eos_token_id = std::stoi(key);
        }

        // if pad_token_id not found use eos_token_id as pad_token_id
        // todo: read m_pad_token_id from tokenizer rt_info once implemented in tokenizers (CVS-144174)
        if (m_pad_token_id == -1 && m_eos_token_id != -1) {
            m_pad_token_id = m_eos_token_id;
        }
    }

    // tokenize str representation to get special tokens integer values
    void infer_special_tokens_if_necessary() {
        auto get_id_from_str = [this](std::string token_str, int64_t& token_val) {
            if (token_val != -1 || token_str.empty())
                return ;
            auto token_ids_tensor = this->encode(token_str).input_ids;
            auto data = token_ids_tensor.data<int64_t>();
            auto data_len = token_ids_tensor.get_shape()[1];
            token_val = data[data_len - 1];
        };
        get_id_from_str(m_pad_token, m_pad_token_id);
        get_id_from_str(m_bos_token, m_bos_token_id);
        get_id_from_str(m_eos_token, m_eos_token_id);
    }

    TokenizedInputs encode(std::string prompt, const ov::AnyMap& tokenization_params = {}) {
        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_tokenizer.get());
        set_state_if_necessary(infer_request_guard, tokenization_params);
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
            CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_tokenizer.get());
            set_state_if_necessary(infer_request_guard, tokenization_params);
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

    std::string decode(std::vector<int64_t> tokens, const ov::AnyMap& detokenization_params = {}) {
        OPENVINO_ASSERT(m_detokenizer, "Detokenize model has not been provided. Tokenizer::decode is not available");

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_detokenizer.get());
        set_state_if_necessary(infer_request_guard, detokenization_params);
        size_t batch_size = 1;
        infer_request_guard.get().set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
        infer_request_guard.get().start_async();
        infer_request_guard.get().wait();
        return infer_request_guard.get().get_output_tensor().data<std::string>()[0];
    }

    std::vector<std::string> decode(ov::Tensor tokens, const ov::AnyMap& detokenization_params = {}) {
        OPENVINO_ASSERT(m_detokenizer, "Detokenize model has not been provided. Tokenizer::decode is not available");
        OPENVINO_ASSERT(tokens.get_element_type() == ov::element::i64, "tokens tensor element type should be an i64");
        OPENVINO_ASSERT(tokens.get_shape().size() == 2, "tokens tensor should of rank 2 with shape [batch_size, seq_len]");

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_detokenizer.get());
        set_state_if_necessary(infer_request_guard, detokenization_params);
        infer_request_guard.get().set_input_tensor(tokens);
        infer_request_guard.get().start_async();
        infer_request_guard.get().wait();

        auto res = infer_request_guard.get().get_output_tensor();
        auto res_data = res.data<std::string>();
        return std::vector<std::string>(res_data, res_data + res.get_shape()[0]);
    }

    std::vector<std::string> decode(std::vector<std::vector<int64_t>> lines, const ov::AnyMap& detokenization_params = {}) {
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
        set_state_if_necessary(infer_request_guard, detokenization_params);
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

    std::string chat_template_from_tokenizer_json_if_exists(const std::filesystem::path& path) {
        auto tokenizer_config_file_path = path / "tokenizer_config.json";
        if (!std::filesystem::exists(tokenizer_config_file_path))
            return "";

        std::ifstream file(tokenizer_config_file_path);
        if (!file.is_open())
            return "";

        std::string res;
        ov::genai::utils::read_json_param(nlohmann::json::parse(file), "chat_template", res);
        
        return patch_chat_template(res);
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

TokenizedInputs Tokenizer::encode(const std::string prompt, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name()});
    return m_pimpl->encode(std::move(prompt), tokenization_params);
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>& prompts, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name()});
    return m_pimpl->encode(prompts, tokenization_params);
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>&& prompts, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name()});
    return m_pimpl->encode(prompts, tokenization_params);
}

TokenizedInputs Tokenizer::encode(std::initializer_list<std::string>& text, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name()});
    return encode(std::vector<std::string>(text.begin(), text.end()), tokenization_params);
}

std::string Tokenizer::decode(std::vector<int64_t> tokens, const ov::AnyMap& detokenization_params) {
    check_arguments(detokenization_params, {ov::genai::skip_special_tokens.name()});
    return m_pimpl->decode(tokens, detokenization_params);
}

std::vector<std::string> Tokenizer::decode(ov::Tensor tokens, const ov::AnyMap& detokenization_params) {
    check_arguments(detokenization_params, {ov::genai::skip_special_tokens.name()});
    return m_pimpl->decode(tokens, detokenization_params);
}

std::vector<std::string> Tokenizer::decode(std::vector<std::vector<int64_t>> lines, const ov::AnyMap& detokenization_params) {
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
