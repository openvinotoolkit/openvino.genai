// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "openvino/genai/llm_pipeline.hpp"
#include <filesystem>
#include <fstream>
#include <variant>
#include "generation_config_helper.hpp"
#include "text_callback_streamer.hpp"
#include "utils.hpp"
#include <nlohmann/json.hpp>

// #include <jinja2cpp/template.h>
// #include <jinja2cpp/template_env.h>
// #include "generation_config.hpp"


namespace ov {

ov::EncodedResults assistive_decoding(ov::InferRequest& m_model_runner, ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig generation_config);
ov::EncodedResults beam_search(ov::InferRequest& model_runner, ov::Tensor prompts, ov::Tensor attentin_mask, GenerationConfig sampling_params);

ov::EncodedResults greedy_decoding(
    ov::InferRequest& model_runner, 
    ov::Tensor prompts, 
    ov::Tensor attentin_mask, 
    GenerationConfig sampling_params, 
    std::shared_ptr<StreamerBase> streamer, 
    bool is_chat_conversation = false
);


class LLMPipeline::LLMPipelineImpl {
public:
    ov::InferRequest m_model_runner;
    Tokenizer m_tokenizer;
    GenerationConfig m_generation_config;
    std::string m_device;
    ov::AnyMap m_plugin_config;
    std::string m_chat_template = "";
    bool is_chat_conversation = false;

    LLMPipelineImpl(
        const std::string model_path,
        const ov::Tokenizer& tokenizer,
        const std::string device,
        const ov::AnyMap& plugin_config
    );

    LLMPipelineImpl(std::string& path, std::string device, const ov::AnyMap& config);
    
    GenerationConfig generation_config() const;

    std::string generate(std::string text, OptionalGenerationConfig generation_config, OptionalStreamerVariant streamer);
    DecodedResults generate(std::vector<std::string> texts, OptionalGenerationConfig generation_config);
    EncodedResults generate(ov::Tensor input_ids, std::optional<ov::Tensor> attention_mask, OptionalGenerationConfig generation_config, OptionalStreamerVariant streamer);

    std::string apply_chat_template(std::string prompt, std::string role = "user") const;
};

} // namespace ov

using namespace std;

std::pair<ov::Tensor, ov::Tensor> pad_left(ov::Tensor&& input_ids, ov::Tensor&& attention_mask, int64_t pad_token) {
    const size_t batch_size = input_ids.get_shape()[0];
    const size_t sequence_length = input_ids.get_shape()[1];
    int64_t* inputs_data = input_ids.data<int64_t>();
    int64_t* attention_mask_data = attention_mask.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;

        // last token in the sequence is not a PAD_TOKEN, skipping
        if (inputs_data[batch_offset + sequence_length - 1] != pad_token)
            continue;

        size_t pad_tokens_number = 0;
        for (int i = sequence_length - 1; i >= 0; i--) {
            const size_t token_offset = batch_offset + i;

            if (inputs_data[token_offset] == pad_token)
                continue;

            if (pad_tokens_number == 0)
                pad_tokens_number = sequence_length - i - 1;

            std::swap(inputs_data[token_offset], inputs_data[token_offset + pad_tokens_number]);
            std::swap(attention_mask_data[token_offset], attention_mask_data[token_offset + pad_tokens_number]);
        }
    }

    return {input_ids, attention_mask};
}

ov::LLMPipeline::LLMPipeline(
    const std::string model_path,
    const ov::Tokenizer& tokenizer,
    const std::string device,
    const ov::AnyMap& plugin_config
) {
    m_pimpl = make_unique<LLMPipelineImpl>(model_path, tokenizer, device, plugin_config);
}

ov::LLMPipeline::LLMPipelineImpl::LLMPipelineImpl(
    const std::string model_path,
    const ov::Tokenizer& tokenizer,
    std::string device,
    const ov::AnyMap& plugin_config
): m_tokenizer(tokenizer), m_device(device), m_plugin_config(plugin_config) {
    ov::Core core;
    
    auto is_xml = [](std::string path) -> bool { return path.compare(path.length() - 4, 4, ".xml") == 0;};
    
    std::string full_path = model_path;
    if (!is_xml(full_path))
        full_path += "/openvino_model.xml";
    try {
        m_model_runner = core.compile_model(full_path, device, plugin_config).create_infer_request();
    } catch (...) {
        OPENVINO_THROW("Cannot compile_model from path " + full_path);
    }
}

ov::LLMPipeline::LLMPipeline(std::string& path, std::string device, const ov::AnyMap& config) {
    m_pimpl = make_unique<LLMPipelineImpl>(path, device, config);
}

ov::LLMPipeline::LLMPipelineImpl::LLMPipelineImpl(std::string& path, std::string device, const ov::AnyMap& config) {
    std::string tokenizer_config_fname = "tokenizer_config.json";
    std::string generation_config_fname = "generation_config.json";

    if (std::filesystem::exists(path + "/" + generation_config_fname)) {
        m_generation_config = GenerationConfig(path + "/" + generation_config_fname);
    }
    if (std::filesystem::exists(path + "/" + tokenizer_config_fname)) {
        std::ifstream f(path + "/" + tokenizer_config_fname);
        nlohmann::json data = nlohmann::json::parse(f);
        m_chat_template = data.value("chat_template", "");
    }
    
    m_device = device;

    ov::Core core;
    auto model_request = core.compile_model(path + "/openvino_model.xml", device, config).create_infer_request();
    m_model_runner = model_request;

    m_tokenizer = Tokenizer(path);
}

ov::GenerationConfig ov::LLMPipeline::LLMPipelineImpl::generation_config() const {
    return m_generation_config;
}

ov::GenerationConfig ov::LLMPipeline::get_generation_config() const {
    return m_pimpl->generation_config();
}

std::string ov::LLMPipeline::LLMPipelineImpl::generate(
    std::string text, 
    OptionalGenerationConfig generation_config,
    OptionalStreamerVariant streamer
) {
    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;

    if (is_chat_conversation) {
        text = apply_chat_template(text);
    }
    auto kv_cache_len = m_model_runner.query_state()[0].get_state().get_shape()[2];
    
    // previous prompt generation in chat dialog stops with the end of sentence token, 
    // need to append this token to the current prompt
    if (is_chat_conversation && kv_cache_len > 0) {
        text = config.eos_token + text;
    }

    auto [input_ids, attention_mask] = m_tokenizer.encode(text);

    // todo: W/A If sentence begins with a special tokens (<bos>, <s>, etc.) openvino_tokenizer inserts 2 special extra tokens <bos> and "▁",
    // but HF does not do that. Moreover openvino_tokenizer always inserts <bos> but in chat scenario HF does not do that because skip_special_tokens=True.
    // Need to remove both of that tokens manually to get exact token by token alignment with HF
    auto size = input_ids.get_shape();
    int64_t* inputs_data = input_ids.data<int64_t>();
    std::vector<int64_t> tmp_ids(inputs_data, inputs_data + input_ids.get_size()); // todo: works only for batch 1
    tmp_ids.erase(tmp_ids.begin());

    auto attention_mask_data = attention_mask.data<int64_t>();
    std::vector<float> tmp_attn_mask(attention_mask_data, attention_mask_data + attention_mask.get_size());
    tmp_attn_mask.erase(tmp_attn_mask.begin());

    std::vector<std::string> prefixes_to_exclude = {"<s>", "</s>"};  // todo: for TinyLlama, need to get them form generation_config
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

    auto generate_results = generate(input_ids, attention_mask, config, streamer);
    return m_tokenizer.decode(generate_results.tokens)[0];
}

ov::DecodedResults ov::LLMPipeline::generate(std::vector<std::string> texts, OptionalGenerationConfig generation_config) {
    return m_pimpl->generate(texts, generation_config);
}

ov::DecodedResults ov::LLMPipeline::LLMPipelineImpl::generate(std::vector<std::string> texts, OptionalGenerationConfig generation_config) {
    auto [input_ids, attention_mask] = m_tokenizer.encode(texts);

    auto generate_results = generate(input_ids, attention_mask, generation_config, {});

    return {m_tokenizer.decode(generate_results.tokens), generate_results.scores};
}

std::string ov::LLMPipeline::operator()(std::string text, OptionalGenerationConfig generation_config) {
    return generate(text, generation_config, {});
}

ov::DecodedResults ov::LLMPipeline::operator()(std::vector<std::string> texts, OptionalGenerationConfig generation_config) {
    return m_pimpl-> generate(texts, generation_config);
}

ov::DecodedResults ov::LLMPipeline::operator()(std::initializer_list<std::string> text, OptionalGenerationConfig generation_config) {
    return m_pimpl->generate(text, generation_config);
}

ov::EncodedResults ov::LLMPipeline::LLMPipeline::generate(ov::Tensor input_ids, 
                                                          std::optional<ov::Tensor> attention_mask, 
                                                          OptionalGenerationConfig generation_config,
                                                          OptionalStreamerVariant streamer) {
    return m_pimpl->generate(input_ids, attention_mask, generation_config, streamer);
}

ov::EncodedResults ov::LLMPipeline::LLMPipelineImpl::generate(
    ov::Tensor input_ids, 
    std::optional<ov::Tensor> attention_mask, OptionalGenerationConfig generation_config, 
    OptionalStreamerVariant streamer
) {
    ov::EncodedResults result;
    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    GenerationConfigHelper config_helper = config;
    
    std::shared_ptr<StreamerBase> streamer_ptr;
    if (!streamer.has_value()){
        streamer_ptr = nullptr;
    } else if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&*streamer)) {
        streamer_ptr = *streamer_obj;
    } else if (auto callback = std::get_if<std::function<void(std::string)>>(&*streamer)) {
        streamer_ptr = std::make_shared<TextCallbackStreamer>(m_tokenizer, *callback);
    }

    auto attention_mask_data = attention_mask.has_value() ? *attention_mask : ov::generate_utils::init_attention_mask(input_ids);

    if (config_helper.is_greedy_decoding()) {
        result = ov::greedy_decoding(m_model_runner, input_ids, attention_mask_data, config, streamer_ptr, is_chat_conversation);
    } else if (config_helper.is_beam_search()) {
        result = beam_search(m_model_runner, input_ids, attention_mask_data, config);
        
    } else if (config_helper.is_multimomial()) {
        // todo: implement multinomial sampling
        // result = multinomial_sampling(input_ids, config);
    } else {
        result = ov::assistive_decoding(m_model_runner, input_ids, attention_mask_data, config);
    }

    if (!is_chat_conversation)
        // reset_state(); todo: implement in m_mimpl
        m_model_runner.reset_state();

    return result;
}

std::string ov::LLMPipeline::generate(std::string text, OptionalGenerationConfig generation_config, OptionalStreamerVariant streamer) {
    return m_pimpl->generate(text, generation_config, streamer);
}


std::string ov::LLMPipeline::generate(std::string text, const ov::AnyMap& config_map) {
    StreamerVariant streamer = {};
    auto config = GenerationConfigHelper(get_generation_config()).anymap_to_generation_config(config_map);

    // todo: get attentions from properties?
    if (config_map.count("streamer_lambda")) {
        streamer = config_map.at("streamer_lambda").as<std::function<void (std::string)>>();
    } else if (config_map.count("streamer")) {
        streamer = config_map.at("streamer").as<std::shared_ptr<StreamerBase>>();
    }

    return m_pimpl->generate(text, config, streamer);
}

std::string ov::LLMPipeline::operator()(std::string text, OptionalGenerationConfig generation_config, StreamerVariant streamer) {
    return m_pimpl->generate(text, generation_config, streamer);
}

std::string ov::LLMPipeline::operator()(std::string text, StreamerVariant streamer) {
    return m_pimpl->generate(text, m_pimpl->m_generation_config, streamer);
}

ov::Tokenizer ov::LLMPipeline::get_tokenizer() {
    return m_pimpl->m_tokenizer;
}

std::string ov::LLMPipeline::apply_chat_template(std::string prompt, std::string role) const {
    return m_pimpl->apply_chat_template(prompt, role);
}

std::string ov::LLMPipeline::LLMPipelineImpl::apply_chat_template(std::string prompt, std::string role) const {
    // todo: temporary disable for easier and faster build
    // jinja2::TemplateEnv env;
    // env.GetSettings().lstripBlocks = true;
    // env.GetSettings().trimBlocks = true;
    // jinja2::Template tpl(&env);
    // tpl.Load(m_chat_template);
    
    // jinja2::ValuesMap message {{"role", role}, {"content", prompt}};
    // jinja2::ValuesMap params = {
    //     {"messages", jinja2::ValuesList({message})},
    //     {"bos_token",  "<s>"},
    //     {"eos_token", "</s>"},  // todo: load from config
    //     {"add_generation_prompt", true},
    // };
 
    // return tpl.RenderAsString(params).value();

    std::stringstream result_prompt;
    result_prompt << "<|user|>\n" << prompt << "</s>\n<|assistant|>\n";  // hardcode template for TinyLlama
    // result_prompt << "<bos><start_of_turn>user\n" << prompt << "<end_of_turn>\n<start_of_turn>model";  // Gemma-7b-it
    // result_prompt << "<s>[INST] " << input << " [/INST]";  // LLama-2-7b
    
    return result_prompt.str();
}

void ov::LLMPipeline::start_chat() {
    m_pimpl->is_chat_conversation = true;
}

void ov::LLMPipeline::finish_chat() {
    m_pimpl->is_chat_conversation = false;
    reset_state();
}

void ov::LLMPipeline::reset_state() {
    m_pimpl->m_model_runner.reset_state();
}

void ov::LLMPipeline::set_generation_config(const GenerationConfig& generation_config) {
    m_pimpl->m_generation_config = generation_config;
}

ov::LLMPipeline::~LLMPipeline() = default;
