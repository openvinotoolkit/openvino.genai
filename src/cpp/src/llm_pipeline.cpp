// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <variant>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <openvino/openvino.hpp>
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "llm_pipeline_base.hpp"
#include "llm_pipeline_static.hpp"
#include "utils.hpp"
#include "text_callback_streamer.hpp"

namespace {

ov::genai::TokenizedInputs subtract_chat_tokenized_inputs(const ov::genai::TokenizedInputs& fisrt, const ov::genai::TokenizedInputs& second){
    auto first_size = fisrt.input_ids.get_size();
    auto second_size = second.input_ids.get_size();
    ov::Shape new_shape{1, first_size - second_size};

    ov::Tensor new_input_ids(ov::element::i64, new_shape);
    auto data_ptr = fisrt.input_ids.data<int64_t>();
    std::copy(data_ptr + second_size, data_ptr + first_size, new_input_ids.data<int64_t>());

    ov::Tensor new_attention_mask(ov::element::i64, new_shape);
    std::fill_n(new_attention_mask.data<int64_t>(), new_shape[1], 1);

    return {new_input_ids, new_attention_mask};
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
    std::optional<ov::Tensor> position_ids
);

ov::genai::EncodedResults multinominal_decoding(
    ov::InferRequest& model_runner,
    ov::Tensor prompts,
    ov::Tensor attention_mask,
    GenerationConfig sampling_params,
    std::shared_ptr<StreamerBase> streamer,
    std::optional<ov::Tensor> position_ids
);

std::pair<EncodedResults, int32_t> beam_search(
    ov::InferRequest& lm, 
    ov::Tensor prompts, 
    ov::Tensor attention_mask, 
    GenerationConfig config,
    std::optional<ov::Tensor> position_ids,
    std::optional<int32_t> selected_beam_idx
);

class StatefulLLMPipeline final : public LLMPipelineImplBase {
public:
    ov::InferRequest m_model_runner;
    
    bool is_chat_conversation = false;
    bool m_is_cache_empty = true;
    std::optional<int32_t> m_selected_beam = std::nullopt;
    ChatHistory m_history;
    std::string m_templated_chat_history = "";

    StatefulLLMPipeline(
        const ov::InferRequest& request,
        const ov::genai::Tokenizer& tokenizer,
        OptionalGenerationConfig generation_config=std::nullopt
    ): LLMPipelineImplBase(tokenizer),
       m_model_runner(request) {
       GenerationConfig default_config;
       m_generation_config = (generation_config.has_value()) ? *generation_config : default_config;
    }

    StatefulLLMPipeline(
        const std::filesystem::path& model_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& plugin_config
    ): 
        LLMPipelineImplBase(tokenizer, utils::from_config_json_if_exists(model_path))
    {
        ov::Core core;
        core.set_property(device, plugin_config);
        m_model_runner = core.compile_model(model_path / "openvino_model.xml", device).create_infer_request();

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1)
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
    }

    StatefulLLMPipeline(
        const std::filesystem::path& model_path, 
        const std::string& device, 
        const ov::AnyMap& plugin_config
    ): StatefulLLMPipeline{model_path, Tokenizer(model_path.string()), device, plugin_config} {}
    
    DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override {
        GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
        EncodedInputs encoded_input;

        if (auto input_vector = std::get_if<std::vector<std::string>>(&inputs)) {
            encoded_input = m_tokenizer.encode(*input_vector);
        } else if (auto input_prompt = std::get_if<std::string>(&inputs)) {
            std::string& prompt = *input_prompt;
            
            if (is_chat_conversation) {
                // KV cache in model already contains prompts and answers from previous iterations.
                // So only new prompt wrapped into chat template to be sent into model. Tokenizer always returns
                // token_ids = {<bos token>, ...<valuable tokens>}. So if tokenizer applies only to the new prompt,
                // <bos token> will be inserted on every iteration.
                // So actual pipeline calculates input_ids for whole chat history + for whole chat history without the new prompt
                // and takes only the difference between them.
                // The chat history cannot be saved as already encoded tokens because generate call doesn't return <eos> token, but
                // KV cache contains it. So we have to add it manually or get it by tokenization all chat history.

                m_history.push_back({{"role", "user"}, {"content", prompt}});
                constexpr bool add_generation_prompt = true;
                auto new_templated_chat_history  = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
                auto new_chat_tokens = m_tokenizer.encode(new_templated_chat_history);
                if (m_is_cache_empty) {
                    encoded_input = new_chat_tokens;
                } else {
                    auto prev_chat_tokens = m_tokenizer.encode(m_templated_chat_history);
                    encoded_input = subtract_chat_tokenized_inputs(new_chat_tokens, prev_chat_tokens);
                }
                m_templated_chat_history = new_templated_chat_history;
            } else {
                encoded_input = m_tokenizer.encode(prompt);
            }
        }

        auto encoded_results  = generate(encoded_input, config, streamer);
        DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
        
        if (is_chat_conversation) {
            // Tail of chat template is missing in KV cache.
            // Find the tail to concatenate it with the next input prompt.
            auto answer = decoded_results.texts[0];
            m_templated_chat_history.append(answer);
            m_history.push_back({{"role", "assistant"}, {"content", answer}});
        }
        
        return decoded_results;
    }

    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override {
        ov::Tensor input_ids;
        ov::Tensor attention_mask;

        if (auto data = std::get_if<ov::Tensor>(&inputs)) {
            input_ids = *data;
            attention_mask = ov::genai::utils::init_attention_mask(input_ids);
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

        
        size_t kv_cache_len = 0;
        ov::Tensor concatenated_attention_mask;
        if (is_chat_conversation && !m_is_cache_empty) {
            OPENVINO_ASSERT(batch_size == 1, "continuation of generation is possible only for batch 1");
            // If history is saved in KV cache, concatenate new attention_mask with the already existing.
            // Between subsequent runs attention_mask should not be modified.
            auto atten_mask_history = m_model_runner.get_tensor("attention_mask");
            auto prompt_len = attention_mask.get_shape()[1];
            kv_cache_len = atten_mask_history.get_shape()[1];

            ov::Tensor new_atten_mask = ov::Tensor{ov::element::i64, {batch_size, kv_cache_len + prompt_len}};
            auto start_atten_hst = atten_mask_history.data<int64_t>() + kv_cache_len * (*m_selected_beam);
            std::copy(start_atten_hst, start_atten_hst + kv_cache_len,
                    new_atten_mask.data<int64_t>());
            std::copy(attention_mask.data<int64_t>(), attention_mask.data<int64_t>() + prompt_len,
                    new_atten_mask.data<int64_t>() + kv_cache_len);
            concatenated_attention_mask = new_atten_mask;
        } else {
            concatenated_attention_mask = attention_mask;
        }

        bool position_ids_available = (num_inputs == 4);
        std::optional<ov::Tensor> position_ids = std::nullopt;
        if (position_ids_available) {
            position_ids = ov::Tensor{ov::element::i64, input_ids.get_shape()};
            utils::initialize_position_ids(*position_ids, attention_mask, kv_cache_len);
        }

        ov::genai::EncodedResults result;
        if (config.is_greedy_decoding()) {
            result = ov::genai::greedy_decoding(m_model_runner, input_ids, concatenated_attention_mask, 
                                                config, streamer_ptr, position_ids);
            m_selected_beam = 0;
        } else if (config.is_beam_search()) {
            std::tie(result, m_selected_beam) = beam_search(m_model_runner, input_ids, concatenated_attention_mask, 
                                                            config, position_ids, m_selected_beam);
        } else if (config.is_multinomial()) {
            result = multinominal_decoding(m_model_runner, input_ids, concatenated_attention_mask, 
                                           config, streamer_ptr, position_ids);
            m_selected_beam = 0;
        } else {
            OPENVINO_THROW("No decoding algorithm found for provided configuration parameters.");
        }

        if (!is_chat_conversation) {
            m_model_runner.reset_state();
            m_selected_beam = std::nullopt;
        } else {
            m_is_cache_empty = false;
        }

        return result;
    }

    void start_chat(const std::string& system_message) override {
        is_chat_conversation = true;
        m_selected_beam  = std::nullopt;
        if (!m_is_cache_empty) {
            m_model_runner.reset_state();
            m_is_cache_empty = true;
            m_history = {};
            m_templated_chat_history = "";
        }
        if (system_message.empty())
            return;

        m_history.push_back({{"role", "system"}, {"content", system_message}});
        constexpr bool add_generation_prompt = false;
        m_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
    }

    void finish_chat() override {
        is_chat_conversation = false;
        m_selected_beam = std::nullopt;
        if (!m_is_cache_empty) {
            m_model_runner.reset_state();
            m_is_cache_empty = true;
            m_history = {};
            m_templated_chat_history = "";
        }
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
    auto config_arg = utils::get_config_from_map(config_map);
    GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return m_pimpl->generate(text, config, utils::get_streamer_from_map(config_map));
}

EncodedResults LLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    return m_pimpl->generate(inputs, generation_config, streamer);
}

EncodedResults LLMPipeline::generate(const EncodedInputs& inputs, const ov::AnyMap& config_map) {
    auto config_arg = utils::get_config_from_map(config_map);
    GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return m_pimpl->generate(inputs, config, utils::get_streamer_from_map(config_map));
}

std::pair<std::string, Any> streamer(StreamerVariant func) {
    if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&func)) {
        return {utils::STREAMER_ARG_NAME, Any::make<std::shared_ptr<StreamerBase>>(*streamer_obj)};
    } else  {
        auto callback = std::get<std::function<bool(std::string)>>(func);
        return {utils::STREAMER_ARG_NAME, Any::make<std::function<bool(std::string)>>(callback)};
    }
}

std::pair<std::string, Any> generation_config(const GenerationConfig& config) {
    return {utils::CONFIG_ARG_NAME, Any::make<GenerationConfig>(config)};
}

}  // namespace genai
}  // namespace ov

using namespace std;

ov::genai::LLMPipeline::LLMPipeline(
    const ov::InferRequest& request,
    const ov::genai::Tokenizer& tokenizer,
    OptionalGenerationConfig generation_config
) {
    m_pimpl = std::make_unique<StatefulLLMPipeline>(request, tokenizer, generation_config);
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::string& model_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& plugin_config
) {
    if (device == "NPU") {
        m_pimpl = make_unique<StaticLLMPipeline>(std::filesystem::path(model_path), tokenizer, device, plugin_config);
    } else {
        m_pimpl = make_unique<StatefulLLMPipeline>(std::filesystem::path(model_path), tokenizer, device, plugin_config);
    }
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::string& path,
    const std::string& device,
    const ov::AnyMap& config
) {
    if (device == "NPU") {
        m_pimpl = make_unique<StaticLLMPipeline>(std::filesystem::path(path), device, config);
    } else {
        m_pimpl = make_unique<StatefulLLMPipeline>(std::filesystem::path(path), device, config);
    }
}

ov::genai::GenerationConfig ov::genai::LLMPipeline::get_generation_config() const {
    return m_pimpl->m_generation_config;
}

ov::genai::Tokenizer ov::genai::LLMPipeline::get_tokenizer() {
    return m_pimpl->m_tokenizer;
}

void ov::genai::LLMPipeline::start_chat(const std::string& system_message) {
    m_pimpl->start_chat(system_message);
}

void ov::genai::LLMPipeline::finish_chat() {
    m_pimpl->finish_chat();
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
