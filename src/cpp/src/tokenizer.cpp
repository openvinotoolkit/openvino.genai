// Copyright (C) 2023-2025 Intel Corporation
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

ov::Core core_with_extension() {
    ov::Core core;
    const char* ov_tokenizer_path = getenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME);
    OPENVINO_ASSERT(ov_tokenizer_path, "openvino_tokenizers path is not set");
    core.add_extension(ov_tokenizer_path);
    return core;
}

ov::Core get_core_singleton() {
    static ov::Core core = core_with_extension();
    return core;
}

const std::pair<std::string, std::string> chat_template_fallback_map[] = {
    {
        // llava-1.5, llava-v1.6-vicuna (chat_template.json)
        "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
        "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'] | upper + ': ' }}{% endif %}{{ message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
    },
    {
        // tiny-random-llava-next, llava-v1.6-mistral (chat_template.json)
        "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'][0]['text'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' }}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{{' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'][0]['text'] + '<\\s> '}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
        "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + '<\\s> ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
    },
    {
        // Qwen2-VL-2B-Instruct (tokenizer_config.json)
        "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    {
        // Qwen2-VL-2B (tokenizer_config.json)
        "{% if messages is string %}{{ messages }}{% else %}{% for content in messages %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}",
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    }
};    

std::optional<std::string> remap_template(const std::string& chat_template) {
    for (const auto& [known, fallback] : chat_template_fallback_map) {
        if (chat_template == known) {
            return fallback;
        }
    }
    return std::nullopt;
}

void parse_if_exists(const std::filesystem::path& path, std::string& value) {
    if (std::filesystem::exists(path)) {
        ov::genai::utils::read_json_param(nlohmann::json::parse(std::ifstream{path}), "chat_template", value);
    }
}

template <typename T>
const T& find_or_fallback(const ov::AnyMap& rt_info, const char name[], const T& fallback) {
    auto iter = rt_info.find(name);
    if (rt_info.end() == iter) {
        return fallback;
    }
    return iter->second.as<T>();
}

std::string patch_template(std::string&& chat_template) {
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
        while ((pos = chat_template.find(from, pos)) != std::string::npos) {
            chat_template.replace(pos, from.size(), to);
            pos += to.size();
        }
    }
    return chat_template;
}

std::string remap_and_patch(const std::string& chat_template) {
    return patch_template(
        remap_template(chat_template).value_or(chat_template)
    );
}

}  // namespace

namespace ov {
namespace genai {

class Tokenizer::TokenizerImpl {
public:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_tokenizer;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_detokenizer;

    // To change the adding special tokens mode we use a statefull subgraph,
    // this flag holds the current state value of the CompiledModel.
    ov::AnyMap m_state_flags;

    bool m_older_than_24_5 = false;

    int64_t m_pad_token_id = -1;
    int64_t m_bos_token_id = -1;
    int64_t m_eos_token_id = -1;

    std::string m_pad_token = {};
    std::string m_bos_token = {};
    std::string m_eos_token = {};

    std::string m_chat_template = {};

    template <typename T>
    void set_state_value(ov::VariableState& state, std::optional<T> value) {
        // better to store which value is in the state locally so that get_state is not called every infer request
        std::optional<T> last_value;
        ov::genai::utils::read_anymap_param(m_state_flags, state.get_name(), last_value);
        
        // If requested add[skip]_special_tokens, max_length, pading mode, etc.
        // is different from the stored state, need to set state variable.
        // Or if we run for the first time and don't know the latest state we need to set it.
        if (value.has_value() && (!last_value.has_value() || *value != *last_value)) {
            ov::Tensor value_tensor = ov::Tensor(ov::element::from<T>(), state.get_state().get_shape());
            OPENVINO_ASSERT(value_tensor.get_size() == 1, "Only flags or single elements values are supported");
            
            *value_tensor.data<T>() = *value;
            state.set_state(value_tensor);
            m_state_flags[state.get_name()] = *value;
        } else if (!value.has_value()) {
            // If user called with params, e.g. tokenizer.encode(prompt, add_special_tokens|max_length=...)
            // After that called without params, e.g. tokenizer.encode(prompt) we should reset to the default state.
            state.reset();
            m_state_flags.erase(state.get_name());
        }
    }

    void set_state_if_necessary(CircularBufferQueueElementGuard<ov::InferRequest>& infer_request_guard, const ov::AnyMap& params) {
        // These values should be equal to default values in py_tokenizer.cpp
        // in order to get the same behavior in C++ when arguments are not specified.
        std::optional<bool> add_special_tokens_flag = true;
        std::optional<bool> skip_special_tokens_flag = true;
        std::optional<int32_t> max_length_val;
        std::optional<bool> pad_to_max_length_val = false;
        
        ov::genai::utils::read_anymap_param(params, add_special_tokens.name(), add_special_tokens_flag);
        ov::genai::utils::read_anymap_param(params, skip_special_tokens.name(), skip_special_tokens_flag);
        ov::genai::utils::read_anymap_param(params, pad_to_max_length.name(), pad_to_max_length_val);
        ov::genai::utils::read_anymap_param(params, max_length.name(), max_length_val);

        if (m_older_than_24_5) {
            // Changing add_special_tokens at runtime was introduced in
            // 24.5. Older tokenizers still allow manipulating their
            // state but the effect is incorrect.
            return;
        }
        
        for (auto& state: infer_request_guard.get().query_state()) {
            auto name = state.get_name();

            if (name == add_special_tokens.name()) {
                set_state_value(state, add_special_tokens_flag);
            } else if (name == skip_special_tokens.name()) {
                set_state_value(state, skip_special_tokens_flag);
            } else if (name == MAX_LENGTH_VAR_ID) {
                set_state_value(state, max_length_val);
            } else if (name == PAD_TO_MAX_LENGTH_VAR_ID) {
                set_state_value(state, pad_to_max_length_val);
            }
        }
    }

    TokenizerImpl(const std::filesystem::path& models_path, const ov::AnyMap& properties) {
        setup_tokenizer(models_path, properties);
    }

    TokenizerImpl(const std::pair<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>>& models, const ov::AnyMap& properties) {
        setup_tokenizer(models, properties);
    }

    void setup_tokenizer(const std::filesystem::path& models_path, const ov::AnyMap& properties) {
        ScopedVar env_manager(tokenizers_relative_to_genai());
        auto core = get_core_singleton();

        OPENVINO_ASSERT(models_path.extension() != ".xml", "'models_path' parameter should be a path to a dir not a xml file");

        std::shared_ptr<ov::Model> ov_tokenizer = nullptr;
        std::shared_ptr<ov::Model> ov_detokenizer = nullptr;

        if (std::filesystem::exists(models_path / "openvino_tokenizer.xml")) {
            ov_tokenizer = core.read_model(models_path / "openvino_tokenizer.xml", {}, properties);
        }

        if (std::filesystem::exists(models_path / "openvino_detokenizer.xml")) {
            ov_detokenizer = core.read_model(models_path / "openvino_detokenizer.xml", {}, properties);
        }

        read_config(models_path);
        read_special_tokens_map(models_path);
        // Try to read tokenizer_config if some token ids or token str are not defined.
        read_tokenizer_config_if_necessary(models_path);
        parse_if_exists(models_path / "tokenizer_config.json", m_chat_template);
        parse_if_exists(models_path / "processor_config.json", m_chat_template);
        parse_if_exists(models_path / "chat_template.json", m_chat_template);
        setup_tokenizer(std::make_pair(ov_tokenizer, ov_detokenizer), properties);
    }

    void setup_tokenizer(const std::pair<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>>& models, const ov::AnyMap& properties) {
        auto [ov_tokenizer, ov_detokenizer] = models;
        OPENVINO_ASSERT(ov_tokenizer || ov_detokenizer, "Neither tokenizer nor detokenzier models were provided");

        auto core = get_core_singleton();
        std::string device = "CPU"; // only CPU is supported for now

        // Saving IR version was added only in 24.5, so if it's missing, then it's older than 24.5
        m_older_than_24_5 = !(ov_tokenizer ? ov_tokenizer: ov_detokenizer)->has_rt_info("openvino_tokenizers_version");

        if (ov_tokenizer) {
            ov::pass::Manager manager;
            manager.register_pass<MakeAddSpecialTokensSatateful>();
            manager.register_pass<MakePaddingSatateful>();
            manager.run_passes(ov_tokenizer);
            ov::CompiledModel tokenizer = core.compile_model(ov_tokenizer, device, properties);
            ov::genai::utils::print_compiled_model_properties(tokenizer, "OV Tokenizer");

            m_ireq_queue_tokenizer = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
                tokenizer.get_property(ov::optimal_number_of_infer_requests),
                [&tokenizer]() -> ov::InferRequest {
                    return tokenizer.create_infer_request();
                });

            const ov::AnyMap& rt_info = ov_tokenizer->get_rt_info();
            m_pad_token_id = find_or_fallback(rt_info, "pad_token_id", m_pad_token_id);
            m_bos_token_id = find_or_fallback(rt_info, "bos_token_id", m_bos_token_id);
            m_eos_token_id = find_or_fallback(rt_info, "eos_token_id", m_eos_token_id);

            m_chat_template = find_or_fallback(rt_info, "chat_template", m_chat_template);
            std::optional<std::string> fallback = remap_template(m_chat_template);
            m_chat_template = patch_template(fallback.value_or(m_chat_template));
            if (!fallback.has_value()) {
                m_chat_template = find_or_fallback(rt_info, "simplified_chat_template", m_chat_template);
            }
            // Initialize tokenizer's cache to save time later.
            // TODO CVS-150630: Empty strings sporadically can fail, therefore use nonempty string for warmup.
            encode("non empty string");
        }

        if (ov_detokenizer) {
            ov::pass::Manager manager_detok;
            manager_detok.register_pass<MakeVocabDecoderSatateful>();
            manager_detok.run_passes(ov_detokenizer);
            ov::CompiledModel detokenizer = core.compile_model(ov_detokenizer, device, properties);
            ov::genai::utils::print_compiled_model_properties(detokenizer, "OV Detokenizer");

            m_ireq_queue_detokenizer = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
                detokenizer.get_property(ov::optimal_number_of_infer_requests),
                [&detokenizer]() -> ov::InferRequest {
                    return detokenizer.create_infer_request();
                });

            // Unset/-1 token causes exception in SentencePiece detokenization.
            if (m_pad_token_id != -1 && m_pad_token.empty())
                m_pad_token = decode(std::vector{m_pad_token_id}, {ov::genai::skip_special_tokens(false)});
            if (m_bos_token_id != -1 && m_bos_token.empty())
                m_bos_token = decode(std::vector{m_bos_token_id}, {ov::genai::skip_special_tokens(false)});
            if (m_eos_token_id != -1 && m_eos_token.empty())
                m_eos_token = decode(std::vector{m_eos_token_id}, {ov::genai::skip_special_tokens(false)});
            // Initialize detokenizer's cache to save time later.
            decode({1, 33, 199, 42, 42});
        }
    }

    // load special tokens ids from config.json
    void read_config(const std::filesystem::path& tokenizer_path) {
        auto config_file_path = tokenizer_path / "config.json";
        if (!std::filesystem::exists(config_file_path))
            return ;
        std::ifstream file(config_file_path);

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

        nlohmann::json data = nlohmann::json::parse(f);

        // they are in the format {"bos_token": { "content": "<s>",... }}
        auto read_token_content_str = [&data](const std::string& key_name, std::string& val) {
            if (val.empty() && data.contains(key_name)) {
                utils::read_json_param(data[key_name], "content", val);
            }
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
        OPENVINO_ASSERT(m_ireq_queue_tokenizer, "Either openvino_tokenizer.xml was not provided or it was not loaded correctly. "
                                                "Tokenizer::encode is not available");

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_tokenizer.get());
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
        OPENVINO_ASSERT(m_ireq_queue_tokenizer, "Either openvino_tokenizer.xml was not provided or it was not loaded correctly. "
                                                "Tokenizer::encode is not available");

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

        return {unpadded.input_ids, unpadded.attention_mask};
    }

    TokenizedInputs get_copied_results(ov::Tensor input_ids, ov::Tensor attention_mask) {
        ov::Tensor input_ids_ = ov::Tensor(input_ids.get_element_type(), input_ids.get_shape());
        ov::Tensor attention_mask_ = ov::Tensor(attention_mask.get_element_type(), attention_mask.get_shape());
        input_ids.copy_to(input_ids_);
        attention_mask.copy_to(attention_mask_);

        return {input_ids_, attention_mask_};
    }

    std::string decode(std::vector<int64_t> tokens, const ov::AnyMap& detokenization_params = {}) {
        OPENVINO_ASSERT(m_ireq_queue_detokenizer, "Detokenizer model has not been provided. Tokenizer::decode is not available");

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_detokenizer.get());
        set_state_if_necessary(infer_request_guard, detokenization_params);
        size_t batch_size = 1;
        infer_request_guard.get().set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
        infer_request_guard.get().start_async();
        infer_request_guard.get().wait();
        return infer_request_guard.get().get_output_tensor().data<std::string>()[0];
    }

    std::vector<std::string> decode(ov::Tensor tokens, const ov::AnyMap& detokenization_params = {}) {
        OPENVINO_ASSERT(m_ireq_queue_detokenizer, "Detokenizer model has not been provided. Tokenizer::decode is not available");
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
        OPENVINO_ASSERT(m_ireq_queue_detokenizer, "Detokenizer model has not been provided. Tokenizer::decode is not available");

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

    std::string apply_chat_template(ChatHistory history,
                                    bool add_generation_prompt,
                                    const std::string& chat_template) const {
        std::string chat_tpl = chat_template.empty() ? m_chat_template : remap_and_patch(chat_template);
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

        std::string result;
        try {
            result = tpl.RenderAsString(params).value();
        } catch (const std::exception& error) {
            OPENVINO_THROW("Chat template for the current model is not supported by Jinja2Cpp. "
                           "Please apply template manually to your prompt before calling generate. "
                           "For example: <start_of_turn>user{user_prompt}<end_of_turn><start_of_turn>model");
        }
        OPENVINO_ASSERT(!result.empty(), "Applied chat template resulted in an empty string. "
                                         "Please check the chat template or apply template manually to your prompt before calling generate."
                                         "For example: <start_of_turn>user{user_prompt}<end_of_turn><start_of_turn>model");
        return result;
    }

    void set_chat_template(const std::string& chat_template) {
        m_chat_template = remap_and_patch(chat_template);
    }

    std::string get_chat_template() {
        return m_chat_template;
    }
};

Tokenizer::Tokenizer(const std::filesystem::path& tokenizer_path, const ov::AnyMap& properties) {
    m_pimpl = std::make_shared<TokenizerImpl>(tokenizer_path, properties);
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
    m_pimpl = std::make_shared<TokenizerImpl>(std::make_pair(ov_tokenizer, ov_detokenizer), properties);
}

Tokenizer::Tokenizer(const std::string& model_str, ov::Tensor& weights_tensor, const ov::AnyMap& properties) {
    ScopedVar env_manager(tokenizers_relative_to_genai());
    auto core = get_core_singleton();
    auto model = core.read_model(model_str, weights_tensor);

    auto parameters = model->get_parameters();
    OPENVINO_ASSERT(!parameters.empty());
    if (parameters.front()->get_element_type() == ov::element::string) {
        // It's a tokenizer
        m_pimpl = std::make_shared<TokenizerImpl>(std::make_pair(model, nullptr), properties);
    } else {
        // It's a detokenizer
        m_pimpl = std::make_shared<TokenizerImpl>(std::make_pair(nullptr, model), properties);
    }
}

TokenizedInputs Tokenizer::encode(const std::string& prompt, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name(), ov::genai::max_length.name(), ov::genai::pad_to_max_length.name()});
    return m_pimpl->encode(prompt, tokenization_params);
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>& prompts, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name(), ov::genai::max_length.name(), ov::genai::pad_to_max_length.name()});
    return m_pimpl->encode(prompts, tokenization_params);
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>&& prompts, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name(), ov::genai::max_length.name(), ov::genai::pad_to_max_length.name()});
    return m_pimpl->encode(prompts, tokenization_params);
}

TokenizedInputs Tokenizer::encode(std::initializer_list<std::string>& text, const ov::AnyMap& tokenization_params) {
    check_arguments(tokenization_params, {ov::genai::add_special_tokens.name(), ov::genai::max_length.name(), ov::genai::pad_to_max_length.name()});
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

std::string Tokenizer::get_chat_template() const {
    return m_pimpl->get_chat_template();
}

void Tokenizer::set_chat_template(const std::string& chat_template) {
    m_pimpl->set_chat_template(chat_template);
}

Tokenizer::~Tokenizer() = default;
}  // namespace genai
}  // namespace ov
