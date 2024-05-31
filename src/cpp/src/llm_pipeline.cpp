// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <variant>

#include <nlohmann/json.hpp>
#include <jinja2cpp/template.h>
#include <jinja2cpp/template_env.h>

#include <openvino/openvino.hpp>
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"
#include "text_callback_streamer.hpp"

namespace {

const std::string STREAMER_ARG_NAME = "streamer";
const std::string CONFIG_ARG_NAME = "generation_config";

ov::genai::GenerationConfig from_config_json_if_exists(const std::string& path) {
    constexpr char generation_config_fname[] = "generation_config.json";
    constexpr char config_fname[] = "config.json";
    if (std::filesystem::exists(path + "/" + generation_config_fname)) {
        return ov::genai::GenerationConfig(path + "/" + generation_config_fname);
    } else if (std::filesystem::exists(path + "/" + config_fname)) {
        // some models (e.g. google/gemma-*) do not have generation_config.json, but have config.json
        // and special tokens are stored there.
        std::ifstream file(path + "/" + config_fname);
        if (!file.is_open())
            return ov::genai::GenerationConfig{};

        nlohmann::json data = nlohmann::json::parse(file);
        using ov::genai::utils::read_json_param;
        ov::genai::GenerationConfig config;

        read_json_param(data, "pad_token_id", config.pad_token_id);
        read_json_param(data, "bos_token_id", config.bos_token_id);
        read_json_param(data, "eos_token_id", config.eos_token_id);
        return config;

    }
    return ov::genai::GenerationConfig{};
}

std::string from_tokenizer_json_if_exists(const std::string& path) {
    std::string res = "";
    
    if (!std::filesystem::exists(path))
        return res;
    
    std::ifstream file(path + "/tokenizer_config.json");
    if (!file.is_open())
        return res;
    
    ov::genai::utils::read_json_param(nlohmann::json::parse(file), "chat_template", res);
    return res;
}

ov::genai::StreamerVariant get_streamer_from_map(const ov::AnyMap& config_map) {
    ov::genai::StreamerVariant streamer = std::monostate();

    if (config_map.count(STREAMER_ARG_NAME)) {
        auto any_val = config_map.at(STREAMER_ARG_NAME);
        if (any_val.is<std::shared_ptr<ov::genai::StreamerBase>>()) {
            streamer = any_val.as<std::shared_ptr<ov::genai::StreamerBase>>();
        } else if (any_val.is<std::function<bool(std::string)>>()) {
            streamer = any_val.as<std::function<bool(std::string)>>();
        }
    }
    return streamer;
}

ov::genai::OptionalGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count(CONFIG_ARG_NAME))
        return config_map.at(CONFIG_ARG_NAME).as<ov::genai::GenerationConfig>();
    else
        return std::nullopt;
}

}

namespace ov {
namespace genai {

ov::genai::EncodedResults greedy_decoding(
    ov::InferRequest& model_runner, 
    ov::Tensor prompts, 
    ov::Tensor attentin_mask, 
    const GenerationConfig sampling_params, 
    const std::shared_ptr<StreamerBase> streamer, 
    const bool is_chat_conversation = false
);

ov::genai::EncodedResults multinominal_decoding(
    ov::InferRequest& model_runner,
    ov::Tensor prompts,
    ov::Tensor attentin_mask,
    GenerationConfig sampling_params,
    std::shared_ptr<StreamerBase> streamer
);

EncodedResults beam_search(ov::InferRequest& lm, ov::Tensor prompts, ov::Tensor attentin_mask, GenerationConfig config);


class LLMPipeline::LLMPipelineImpl {
public:
    ov::InferRequest m_model_runner;
    Tokenizer m_tokenizer;
    GenerationConfig m_generation_config;
    std::string m_chat_template = "";
    bool is_chat_conversation = false;
    bool m_is_cache_empty = true;

    LLMPipelineImpl(
        const ov::InferRequest& request, 
        const ov::genai::Tokenizer& tokenizer, 
        OptionalGenerationConfig generation_config=std::nullopt
    ): m_model_runner(request), m_tokenizer(tokenizer) {
        GenerationConfig default_config;
        m_generation_config = (generation_config.has_value()) ? *generation_config : default_config;
    }

    LLMPipelineImpl(
        const std::string& model_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& plugin_config
    );

    LLMPipelineImpl(
        const std::string& path, 
        const std::string& device, 
        const ov::AnyMap& config
    );
    
    DecodedResults generate(
        StringInputs inputs, 
        OptionalGenerationConfig generation_config, 
        StreamerVariant streamer
    ) {
        GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
        
        auto old_pad_token_id = m_tokenizer.get_pad_token_id();
        m_tokenizer.set_pad_token_id(config.pad_token_id);

        EncodedInputs encoded_input;
        if (auto input_vector = std::get_if<std::vector<std::string>>(&inputs)) {
            encoded_input = m_tokenizer.encode(*input_vector);
        } else if (auto input_str = std::get_if<std::string>(&inputs)) {
            
            std::string text = *input_str;
            // todo: make for batched inputs as well
            if (is_chat_conversation)
                text = apply_chat_template(text);

            // previous prompt generation in chat dialog stops with the end of sentence token, 
            // need to append this token to the current prompt
            if (is_chat_conversation && !m_is_cache_empty)
                text = config.eos_token + text;
            
            auto res = m_tokenizer.encode(text);
            auto input_ids = res.input_ids;
            auto attention_mask = res.attention_mask;

            // todo: W/A If sentence begins with a specfial tokens (<bos>, <s>, etc.) openvino_tokenizer inserts 2 special extra tokens <bos> and "▁",
            // but HF does not do that. Moreover openvino_tokenizer always inserts <bos> but in chat scenario HF does not do that because skip_special_tokens=True.
            // Need to remove both of that tokens manually to get exact token by token alignment with HF
            auto size = input_ids.get_shape();
            int64_t* inputs_data = input_ids.data<int64_t>();
            std::vector<int64_t> tmp_ids(inputs_data, inputs_data + input_ids.get_size()); // todo: works only for batch 1
            // tmp_ids.erase(tmp_ids.begin());

            auto attention_mask_data = attention_mask.data<int64_t>();
            std::vector<float> tmp_attn_mask(attention_mask_data, attention_mask_data + attention_mask.get_size());
            // tmp_attn_mask.erase(tmp_attn_mask.begin());

            std::vector<std::string> prefixes_to_exclude = {config.eos_token, config.bos_token};
            auto prefix_match = [&text](std::string prefix) { return text.substr(0, prefix.length()) == prefix; };
            if (std::any_of(prefixes_to_exclude.begin(), prefixes_to_exclude.end(), prefix_match)) {
                tmp_ids.erase(tmp_ids.begin());
                tmp_attn_mask.erase(tmp_attn_mask.begin());
            }

            input_ids = ov::Tensor(input_ids.get_element_type(), {1, tmp_ids.size()});
            for (size_t i = 0; i < tmp_ids.size(); i++)
                input_ids.data<int64_t>()[i] = tmp_ids.data()[i];
            attention_mask = ov::Tensor(attention_mask.get_element_type(), {1, tmp_attn_mask.size()});
            for (size_t i = 0; i < tmp_attn_mask.size(); i++)
                attention_mask.data<int64_t>()[i] = tmp_attn_mask.data()[i];
            
            encoded_input = TokenizedInputs{input_ids, attention_mask};
        }
        m_tokenizer.set_pad_token_id(old_pad_token_id);
        auto encoded_results  = generate(encoded_input, config, streamer);
        return {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    }

    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config, 
        StreamerVariant streamer
    ) {
        ov::Tensor input_ids;
        ov::Tensor attention_mask;

        // input_ids
        // attention_mask
        if (auto data = std::get_if<ov::Tensor>(&inputs)) {
            input_ids = *data;
        } else if (auto data = std::get_if<std::pair<ov::Tensor, ov::Tensor>>(&inputs)) {
            input_ids = data->first;
            attention_mask = data->second;
        } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
            input_ids = data->input_ids;
            attention_mask = data->attention_mask;
        }
        
        ov::genai::EncodedResults result;
        GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
        
        std::shared_ptr<StreamerBase> streamer_ptr;

        if (auto streamer_obj = std::get_if<std::monostate>(&streamer)) {
            streamer_ptr = nullptr;
        } else if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
            streamer_ptr = *streamer_obj;
        } else if (auto callback = std::get_if<std::function<bool(std::string)>>(&streamer)) {
            streamer_ptr = std::make_shared<TextCallbackStreamer>(m_tokenizer, *callback);
        }

        auto batch_size = input_ids.get_shape().at(0);
        if ((batch_size != 1 || !(config.is_greedy_decoding() || config.is_multinomial())) && streamer_ptr) {
            OPENVINO_THROW("Currently streaming is possible only with batch size=1 and greedy or multinomial decoding");
        }

        if (config.is_greedy_decoding()) {
            result = ov::genai::greedy_decoding(m_model_runner, input_ids, attention_mask, config, streamer_ptr, is_chat_conversation);
        } else if (config.is_beam_search()) {
            result = beam_search(m_model_runner, input_ids, attention_mask, config);
        } else if (config.is_multinomial()) {
            result = multinominal_decoding(m_model_runner, input_ids, attention_mask, config, streamer_ptr);
        } else {
            OPENVINO_THROW("No decoding algorithm found for provided configuration parameters.");
        }

        if (!is_chat_conversation)
            m_model_runner.reset_state();
        
        m_is_cache_empty = false;
        return result;        
    }

    std::string apply_chat_template(std::string prompt, std::string role = "user") const {
        jinja2::TemplateEnv env;
        env.GetSettings().lstripBlocks = true;
        env.GetSettings().trimBlocks = true;
        jinja2::Template tpl(&env);
        tpl.Load(m_chat_template);
        
        jinja2::ValuesMap message {{"role", role}, {"content", prompt}};
        jinja2::ValuesMap params = {
            {"messages", jinja2::ValuesList({message})},
            {"bos_token",  m_generation_config.bos_token},
            {"eos_token", m_generation_config.eos_token},
            {"add_generation_prompt", true},
        };
    
        return tpl.RenderAsString(params).value();
    }

    std::vector<std::string> apply_chat_template(std::vector<std::string>& prompts, std::string role = "user") const {
        std::vector<std::string> res;
        for (const auto& prompt: prompts) {
            res.emplace_back(apply_chat_template(prompt));
        }
        return res;
    }
};

DecodedResults LLMPipeline::generate(
        StringInputs inputs, 
        OptionalGenerationConfig generation_config, 
        StreamerVariant streamer
) {
    return m_pimpl->generate(inputs, generation_config, streamer);
}

DecodedResults LLMPipeline::generate(StringInputs text, const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return m_pimpl->generate(text, config, get_streamer_from_map(config_map));
}

EncodedResults LLMPipeline::generate(
    const EncodedInputs& inputs, 
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    return m_pimpl->generate(inputs, generation_config, streamer);
}

EncodedResults LLMPipeline::generate(const EncodedInputs& inputs, const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return m_pimpl->generate(inputs, config, get_streamer_from_map(config_map));
}

std::pair<std::string, Any> streamer(StreamerVariant func) {
    if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&func)) {
        return {STREAMER_ARG_NAME, Any::make<std::shared_ptr<StreamerBase>>(*streamer_obj)};
    } else  {
        auto callback = std::get<std::function<bool(std::string)>>(func);
        return {STREAMER_ARG_NAME, Any::make<std::function<bool(std::string)>>(callback)};
    } 
}

std::pair<std::string, Any> generation_config(const GenerationConfig& config) {
    return {CONFIG_ARG_NAME, Any::make<GenerationConfig>(config)};
}

}  // namespace genai
}  // namespace ov

using namespace std;

ov::genai::LLMPipeline::LLMPipeline(
    const ov::InferRequest& request, 
    const ov::genai::Tokenizer& tokenizer, 
    OptionalGenerationConfig generation_config
) {
    m_pimpl = std::make_unique<LLMPipelineImpl>(request, tokenizer, generation_config);
}


ov::genai::LLMPipeline::LLMPipeline(
    const std::string& model_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& plugin_config
) {
    m_pimpl = make_unique<LLMPipelineImpl>(model_path, tokenizer, device, plugin_config);
}

ov::genai::LLMPipeline::LLMPipelineImpl::LLMPipelineImpl(
    const std::string& model_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& plugin_config
): m_tokenizer(tokenizer) {
    ov::Core core;
    
    std::string full_path = model_path;
    if (!ov::genai::utils::is_xml(full_path))
        full_path += "/openvino_model.xml";
    try {
        m_model_runner = core.compile_model(full_path, device, plugin_config).create_infer_request();
    } catch (...) {
        OPENVINO_THROW("Cannot compile_model from path " + full_path);
    }
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::string& path, 
    const std::string& device, 
    const ov::AnyMap& config
) {
    m_pimpl = make_unique<LLMPipelineImpl>(path, device, config);
}

ov::genai::LLMPipeline::LLMPipelineImpl::LLMPipelineImpl(
    const std::string& path, 
    const std::string& device, 
    const ov::AnyMap& config
): 
    m_model_runner{ov::Core{}.compile_model(path + "/openvino_model.xml", device, config).create_infer_request()}, 
    m_tokenizer(path, device),
    m_generation_config{from_config_json_if_exists(path)},
    m_chat_template{from_tokenizer_json_if_exists(path)}
 {
 }

ov::genai::GenerationConfig ov::genai::LLMPipeline::get_generation_config() const {
    return m_pimpl->m_generation_config;
}

ov::genai::Tokenizer ov::genai::LLMPipeline::get_tokenizer() {
    return m_pimpl->m_tokenizer;
}

std::string ov::genai::LLMPipeline::apply_chat_template(std::string prompt, std::string role) const {
    return m_pimpl->apply_chat_template(prompt, role);
}


void ov::genai::LLMPipeline::start_chat() {
    m_pimpl->is_chat_conversation = true;
    if (!m_pimpl->m_is_cache_empty) {
        m_pimpl->m_model_runner.reset_state();
        m_pimpl->m_is_cache_empty = true;
    }
}

void ov::genai::LLMPipeline::finish_chat() {
    m_pimpl->is_chat_conversation = false;
    if (!m_pimpl->m_is_cache_empty) {
        m_pimpl->m_model_runner.reset_state();
        m_pimpl->m_is_cache_empty = true;
    }
}

void ov::genai::LLMPipeline::set_generation_config(const GenerationConfig& generation_config) {
    m_pimpl->m_generation_config = generation_config;
}

ov::genai::LLMPipeline::~LLMPipeline() = default;
