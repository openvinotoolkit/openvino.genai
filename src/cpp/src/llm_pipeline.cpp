// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <variant>
#include <algorithm>

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

ov::genai::GenerationConfig from_config_json_if_exists(const std::filesystem::path& model_path) {
    auto config_file_path = model_path / "generation_config.json";
    if (std::filesystem::exists(config_file_path)) {
        return ov::genai::GenerationConfig((config_file_path).string());
    } else {
        return ov::genai::GenerationConfig{};
    }
}

std::string chat_template_from_tokenizer_json_if_exists(const std::filesystem::path& path) {
    auto tokenizer_config_file_path = path / "tokenizer_config.json";
    if (!std::filesystem::exists(tokenizer_config_file_path))
        return "";
    
    std::ifstream file(tokenizer_config_file_path);
    if (!file.is_open())
        return "";
    
    std::string res = "";
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
    ov::Tensor attention_mask, 
    const GenerationConfig sampling_params, 
    const std::shared_ptr<StreamerBase> streamer, 
    const bool is_chat_conversation = false,
    const bool is_cache_empty = true
);

ov::genai::EncodedResults multinominal_decoding(
    ov::InferRequest& model_runner,
    ov::Tensor prompts,
    ov::Tensor attention_mask,
    GenerationConfig sampling_params,
    std::shared_ptr<StreamerBase> streamer
);

EncodedResults beam_search(
    ov::InferRequest& lm, 
    ov::Tensor prompts, 
    ov::Tensor attention_mask, 
    GenerationConfig config
);


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
    ):  
        m_model_runner(request), 
        m_tokenizer(tokenizer) {
        GenerationConfig default_config;
        m_generation_config = (generation_config.has_value()) ? *generation_config : default_config;
    }

    LLMPipelineImpl(
        const std::filesystem::path& model_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& plugin_config
    ): 
         m_model_runner{
            ov::Core{}.compile_model(
                model_path / "openvino_model.xml", 
                device, 
                plugin_config
            ).create_infer_request()
        },
        m_tokenizer(tokenizer),
        m_generation_config{from_config_json_if_exists(model_path)},
        m_chat_template{chat_template_from_tokenizer_json_if_exists(model_path)}
    {
        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1)
            m_generation_config.eos_token_id = m_tokenizer.get_eos_token_id();
    }

    LLMPipelineImpl(
        const std::filesystem::path& model_path, 
        const std::string& device, 
        const ov::AnyMap& plugin_config
    ): 
         m_model_runner{
            ov::Core{}.compile_model(
                model_path / "openvino_model.xml", 
                device, 
                plugin_config
            ).create_infer_request()
        }, 
        m_tokenizer(model_path.string()),
        m_generation_config{from_config_json_if_exists(model_path)},
        m_chat_template{chat_template_from_tokenizer_json_if_exists(model_path)}
    {
        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1)
            m_generation_config.eos_token_id = m_tokenizer.get_eos_token_id();
    }
    
    DecodedResults generate(
        StringInputs inputs, 
        OptionalGenerationConfig generation_config, 
        StreamerVariant streamer
    ) {
        GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
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
                text = m_tokenizer.get_eos_token() + text;
            
            auto res = m_tokenizer.encode(text);
            auto input_ids = res.input_ids;
            auto attention_mask = res.attention_mask;

            // todo: W/A If sentence begins with specfial tokens (<bos>, <s>, etc.) openvino_tokenizer inserts 2 special extra tokens <bos> and "▁",
            // but HF does not do that. Moreover openvino_tokenizer always inserts <bos> but in chat scenario HF does not do that because skip_special_tokens=True.
            // Need to remove both of that tokens manually to get exact token by token alignment with HF
            auto size = input_ids.get_shape();
            int64_t* inputs_data = input_ids.data<int64_t>();
            std::vector<int64_t> tmp_ids(inputs_data, inputs_data + input_ids.get_size()); // todo: works only for batch 1

            auto attention_mask_data = attention_mask.data<int64_t>();
            std::vector<float> tmp_attn_mask(attention_mask_data, attention_mask_data + attention_mask.get_size());

            std::vector<std::string> prefixes_to_exclude = {m_tokenizer.get_eos_token(), m_tokenizer.get_bos_token()};
            auto prefix_match = [&text](std::string prefix) { return text.substr(0, prefix.length()) == prefix; };
            if (std::any_of(prefixes_to_exclude.begin(), prefixes_to_exclude.end(), prefix_match)) {
                tmp_ids.erase(tmp_ids.begin());
                tmp_attn_mask.erase(tmp_attn_mask.begin());
            }
            
            input_ids = ov::Tensor(input_ids.get_element_type(), {1, tmp_ids.size()});
            attention_mask = ov::Tensor(attention_mask.get_element_type(), {1, tmp_attn_mask.size()});
            std::copy(tmp_ids.begin(), tmp_ids.end(), input_ids.data<int64_t>());
            std::copy(tmp_attn_mask.begin(), tmp_attn_mask.end(), attention_mask.data<int64_t>());
           
            encoded_input = TokenizedInputs{input_ids, attention_mask};
        }

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

        if (auto data = std::get_if<ov::Tensor>(&inputs)) {
            input_ids = *data;
        } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
            input_ids = data->input_ids;
            attention_mask = data->attention_mask;
        }
        
        GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
        
        // If eos_token_id was not provided, take value from default m_generation_config
        if (config.eos_token_id == -1)
            config.eos_token_id = m_generation_config.eos_token_id;
        config.validate();
        
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
            OPENVINO_THROW("Currently streaming is possible only with batch size=1 and "
                            "only for greedy or multinomial decoding");
        }

        auto num_inputs = m_model_runner.get_compiled_model().inputs().size();
        OPENVINO_ASSERT(num_inputs == 4 || num_inputs == 3, "Model should have 3 or 4 inputs: "
                        "either (input_ids, attention_mask, beam_idx) or "
                        "(input_ids, attention_mask, position_ids, beam_idx) "
                        "but you have '" + std::to_string(num_inputs) + "' inputs");

        ov::genai::EncodedResults result;
        if (config.is_greedy_decoding()) {
            result = ov::genai::greedy_decoding(m_model_runner, input_ids, attention_mask, 
                                                config, streamer_ptr, 
                                                is_chat_conversation, m_is_cache_empty);
        } else if (config.is_beam_search()) {
            result = beam_search(m_model_runner, input_ids, attention_mask, config);
        } else if (config.is_multinomial()) {
            result = multinominal_decoding(m_model_runner, input_ids, attention_mask, config, streamer_ptr);
        } else {
            OPENVINO_THROW("No decoding algorithm found for provided configuration parameters.");
        }

        if (!is_chat_conversation) {
            m_model_runner.reset_state();
        } else {
            m_is_cache_empty = false;
        }
        
        return result;        
    }

    std::string apply_chat_template(const std::vector<std::pair<std::string, std::string>>& prompts) const {
        jinja2::TemplateEnv env;
        env.GetSettings().lstripBlocks = true;
        env.GetSettings().trimBlocks = true;
        jinja2::Template tpl(&env);
        tpl.Load(m_chat_template);

        jinja2::ValuesList messages;
        for (const auto& [prompt, role] : prompts) {
            messages.push_back(jinja2::ValuesMap{{"role", role}, {"content", prompt}});
        }

        jinja2::ValuesMap params = {
            {"messages", messages},
            {"bos_token", m_tokenizer.get_bos_token()},
            {"eos_token", m_tokenizer.get_eos_token()},
            {"add_generation_prompt", true},
        };

        return tpl.RenderAsString(params).value();
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
            {"bos_token",  m_tokenizer.get_bos_token()},
            {"eos_token", m_tokenizer.get_eos_token()},
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
    m_pimpl = make_unique<LLMPipelineImpl>(std::filesystem::path(model_path), tokenizer, device, plugin_config);
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::string& path, 
    const std::string& device, 
    const ov::AnyMap& config
) {
    m_pimpl = make_unique<LLMPipelineImpl>(std::filesystem::path(path), device, config);
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

void ov::genai::LLMPipeline::set_generation_config(const GenerationConfig& config) {
    int64_t default_eos_token_id = m_pimpl->m_generation_config.eos_token_id;;
    m_pimpl->m_generation_config = config;
    // if eos_token_id was not provided in config forward from default config
    if (config.eos_token_id == -1)
        m_pimpl->m_generation_config.eos_token_id = default_eos_token_id;
    
    m_pimpl->m_generation_config.validate();
}

ov::genai::LLMPipeline::~LLMPipeline() = default;
