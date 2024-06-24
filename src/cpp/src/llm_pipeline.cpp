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
#include "utils.hpp"
#include "text_callback_streamer.hpp"

#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

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

class LLMPipelineImplBase {
public:
    LLMPipelineImplBase(const Tokenizer& tokenzer,
                        const GenerationConfig& config = {});

    virtual DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) = 0;

    virtual EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) = 0;

    virtual void start_chat() = 0;
    virtual void finish_chat() = 0;

    virtual ~LLMPipelineImplBase() = default;

    Tokenizer m_tokenizer;
    GenerationConfig m_generation_config;
};

LLMPipelineImplBase::LLMPipelineImplBase(const Tokenizer& tokenizer,
                                         const GenerationConfig& config)
    : m_tokenizer(tokenizer), m_generation_config(config) {
}

class LLMPipelineImpl final : public LLMPipelineImplBase {
public:
    ov::InferRequest m_model_runner;
    
    bool is_chat_conversation = false;
    bool m_is_cache_empty = true;
    ChatHistory m_history;
    std::string m_templated_chat_history = "";

    LLMPipelineImpl(
        const ov::InferRequest& request,
        const ov::genai::Tokenizer& tokenizer,
        OptionalGenerationConfig generation_config=std::nullopt
    ): LLMPipelineImplBase(tokenizer),
       m_model_runner(request) {
       GenerationConfig default_config;
       m_generation_config = (generation_config.has_value()) ? *generation_config : default_config;
    }

    LLMPipelineImpl(
        const std::filesystem::path& model_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& plugin_config
    ): 
        LLMPipelineImplBase(tokenizer, from_config_json_if_exists(model_path))
    {
        ov::Core core;
        core.set_property(device, plugin_config);
        m_model_runner = core.compile_model(model_path / "openvino_model.xml", device).create_infer_request();

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1)
            m_generation_config.eos_token_id = m_tokenizer.get_eos_token_id();
    }

    LLMPipelineImpl(
        const std::filesystem::path& model_path, 
        const std::string& device, 
        const ov::AnyMap& plugin_config
    ): LLMPipelineImpl{model_path, Tokenizer(model_path.string()), device, plugin_config} {}
    
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
                m_history.push_back({{"role", "user"}, {"content", prompt}});
                constexpr bool add_generation_prompt = true;
                auto new_templated_chat_history  = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
                
                prompt = new_templated_chat_history.substr(m_templated_chat_history.size());
                m_templated_chat_history = new_templated_chat_history;
            }
            
            encoded_input = m_tokenizer.encode(prompt);
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

    void start_chat() override {
        is_chat_conversation = true;
        if (!m_is_cache_empty) {
            m_model_runner.reset_state();
            m_is_cache_empty = true;
        }
    }

    void finish_chat() override {
        is_chat_conversation = false;
        if (!m_is_cache_empty) {
            m_model_runner.reset_state();
            m_is_cache_empty = true;
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

namespace {

std::shared_ptr<ov::Model> add_slices_to_kvcache_inputs(const std::shared_ptr<ov::Model>& model) {
    const auto kvcache_name_pattern = "past_key_values";
    std::vector<std::shared_ptr<ov::opset13::Parameter>> new_params;
    for (auto param : model->get_parameters()) {
        auto tensor_name = param->get_output_tensor(0).get_any_name();
        if (tensor_name.find(kvcache_name_pattern) == std::string::npos) {
            new_params.push_back(param);
            continue;
        }
        auto shape = param->get_output_shape(0);
        shape[2] += 1;

        auto new_param = std::make_shared<ov::opset13::Parameter>(param->get_element_type(), shape);
        new_param->set_friendly_name(tensor_name);
        new_param->outputs().begin()->get_tensor().set_names(param->outputs().begin()->get_tensor().get_names());

        auto slice_start = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{1}
        );
        auto slice_stop = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{static_cast<int32_t>(shape[2])}
        );
        auto slice_step = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{1}
        );
        auto slice_axes = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{2}
        );
        auto slice_node = std::make_shared<ov::opset13::Slice>(
            new_param, slice_start->output(0), slice_stop->output(0), slice_step->output(0), slice_axes->output(0)
        );
        slice_node->set_friendly_name(tensor_name + "_Slice");
        for (auto target_input : param->output(0).get_target_inputs()) {
            target_input.replace_source_output(slice_node->output(0));
        }
        new_params.push_back(new_param);
    }
    return std::make_shared<ov::Model>(model->get_results(), ov::SinkVector{}, new_params);
}

void reshape_to_static(std::shared_ptr<ov::Model> model,
                       const uint32_t input_size,
                       const uint32_t kvcache_size) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("attention_mask") != std::string::npos) {
            new_shape = ov::PartialShape({1, kvcache_size});
        } else if (input_name.find("position_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = ov::PartialShape({1,
                                          partial_shape[1].get_length(),
                                          kvcache_size-input_size,
                                          partial_shape[3].get_length()});
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

void fill_tensor(ov::Tensor tensor, int64_t fill_val) {
    int64_t* tensor_data = tensor.data<int64_t>();
    std::fill(tensor_data, tensor_data + tensor.get_size(), fill_val);
}

void copy_with_left_offset(const ov::Tensor& orig, ov::Tensor& padded) {
    const auto orig_size = orig.get_size();
    const auto padded_size = padded.get_size();
    const auto kLeftOffset = padded_size - orig_size;
    int64_t* orig_data = orig.data<int64_t>();
    int64_t* padded_data = padded.data<int64_t>();
    std::copy(orig_data, orig_data + orig_size, padded_data + kLeftOffset);
}

} // anonymous namespace

namespace ov {
namespace genai {

class NPULLMPipelineImpl final : public LLMPipelineImplBase {
public:
    NPULLMPipelineImpl(
        const std::filesystem::path& path,
        const ov::genai::Tokenizer& tokenizer,
        const ov::AnyMap& config
    );

    NPULLMPipelineImpl(
        const std::filesystem::path& path,
        const ov::AnyMap& config
    );

    DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    void start_chat() override {
        OPENVINO_THROW("Currently chat conversation mode isn't supported for NPU device");
    };
    void finish_chat() override {
        OPENVINO_THROW("Currently chat conversation mode isn't supported for NPU device");
    };

private:
    void prepare_for_new_conversation();

private:
    struct KVCacheDesc {
        uint32_t total_size;
        uint32_t num_stored_tokens;
    };

    KVCacheDesc m_kvcache_desc;
    ov::InferRequest m_kvcache_request;
    ov::InferRequest m_prefill_request;
};

NPULLMPipelineImpl::NPULLMPipelineImpl(
    const std::filesystem::path& path,
    const ov::genai::Tokenizer& tokenizer,
    const ov::AnyMap& config
) : LLMPipelineImplBase(tokenizer,
                        from_config_json_if_exists(path)) {
    /* NB: NPU-friendly LLM pipeline consists of two models,
       first to process the input prompt (prefill), second to use in generation loop (kvcache)

       Initialization assumes multiple steps:
       1) Read the template model and clone it for the further use as kvcache and prefill models
       2) Expose KV-cache input and output layers for the kvcache model
       3) Reshape both models to static shape
       4) Add slices to KV-cache inputs for kvcache model, this will make input and output KV-cache
          layers to have the same shape and allow outputs writes directly to inputs for the next iteration.
       5) Compile both models for NPUW
       6) Initialize input tensors for kvcache and prefill models
    */

    ov::Core core;
    // (1) Read the template model - this will be kvcache
    auto kvcache_model = core.read_model(path / "openvino_model.xml");
    // (2) Expose KV-cache input and output layers
    ov::pass::StatefulToStateless().run_on_model(kvcache_model);
    // (3) Clone the model - this will be prefill
    auto prefill_model = kvcache_model->clone();
    prefill_model->set_friendly_name(kvcache_model->get_friendly_name() + "_prefill");
    // (4) Reshape both models to static shape
    // FIXME: There must be better logic than just hardcoded values
    m_kvcache_desc = KVCacheDesc { 1024u, 0u };
    const uint32_t max_prompt_size = m_kvcache_desc.total_size;
    const uint32_t max_kvcache_size = m_kvcache_desc.total_size;
    reshape_to_static(prefill_model, max_prompt_size, max_kvcache_size);
    reshape_to_static(kvcache_model, 1u, max_kvcache_size);
    // (5) Add slices to kvcache model
    kvcache_model = add_slices_to_kvcache_inputs(kvcache_model);
    // (6) Compiled both models for NPUW
    std::map<std::string, std::string> cfg = { {"NPU_USE_NPUW", "YES"},
                                               {"NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add" },
                                               {"NPUW_FOLD", "YES"},
                                               {"NPUW_DCOFF_TYPE", "f16"},
                                               {"NPUW_ONLINE_PIPELINE", "NONE"},
                                               {"NPUW_DCOFF_SCALE", "YES"} };
    ov::AnyMap properties{cfg.begin(), cfg.end()};
    m_prefill_request = core.compile_model(prefill_model, "NPU", properties).create_infer_request();
    m_kvcache_request = core.compile_model(kvcache_model, "NPU", properties).create_infer_request();
    // (7) Initialize tensors
    prepare_for_new_conversation();
};

NPULLMPipelineImpl::NPULLMPipelineImpl(
    const std::filesystem::path& path,
    const ov::AnyMap& config
) : NPULLMPipelineImpl(path, path.string(), config) {
}

void NPULLMPipelineImpl::prepare_for_new_conversation() {
    fill_tensor(m_prefill_request.get_tensor("input_ids"), m_tokenizer.get_pad_token_id());
    fill_tensor(m_prefill_request.get_tensor("position_ids"), 0u);
    fill_tensor(m_prefill_request.get_tensor("attention_mask"), 0u);
    fill_tensor(m_kvcache_request.get_tensor("attention_mask"), 0u);
    m_kvcache_desc.num_stored_tokens = 0u;
}

DecodedResults NPULLMPipelineImpl::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    if (std::holds_alternative<std::vector<std::string>>(inputs)) {
        OPENVINO_THROW("Currently only batch size=1 is supported for NPU device");
    }

    OPENVINO_ASSERT(std::holds_alternative<std::string>(inputs));
    const auto& text = std::get<std::string>(inputs);
    auto tokenized_input = m_tokenizer.encode(text);
    auto encoded_results = generate(tokenized_input, config, streamer);
    return {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
}

EncodedResults NPULLMPipelineImpl::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    ov::Tensor input_ids;
    ov::Tensor attention_mask;

    if (auto data = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *data;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = data->input_ids;
        attention_mask = data->attention_mask;
    }

    if (input_ids.get_shape().at(0) > 1u) {
        OPENVINO_THROW("Currently only batch size=1 is supported for NPU device");
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

    if (!config.is_greedy_decoding()) {
        OPENVINO_THROW("Currently only greedy decoding is supported for NPU device");
    }

    ov::genai::EncodedResults results;
    // NB: Only batch=1 is supported now
    results.scores.resize(1u);
    results.tokens.resize(1u);

    // NB: Check if input prompt less than maximum size supported for NPU device
    auto prompt_len = input_ids.get_size();
    if (prompt_len > m_kvcache_desc.total_size) {
        OPENVINO_THROW("Currently NPU device may only process up to " + std::to_string(m_kvcache_desc.total_size) + " tokens");
    }

    // NB: Reset tensors on every generate call - chat conversation isn't supported yet!
    prepare_for_new_conversation();

    auto padded_input_ids = m_prefill_request.get_tensor("input_ids");
    copy_with_left_offset(input_ids, padded_input_ids);

    auto padded_attention_mask = m_prefill_request.get_tensor("attention_mask");
    copy_with_left_offset(attention_mask, padded_attention_mask);

    auto padded_position_ids = m_prefill_request.get_tensor("position_ids");
    auto* padded_pos_data = padded_position_ids.data<int64_t>();
    std::iota(padded_pos_data + (m_kvcache_desc.total_size - prompt_len + 1), padded_pos_data + padded_position_ids.get_size(), 0u);

    m_prefill_request.infer();

    // NB: Now there are prompt_len tokens in KV-cache
    m_kvcache_desc.num_stored_tokens += prompt_len;
    int64_t last_token = utils::argmax(m_prefill_request.get_tensor("logits"), 0);
    if (streamer_ptr && streamer_ptr->put(last_token)) {
        return results;
    }

    padded_attention_mask.copy_to(m_kvcache_request.get_tensor("attention_mask"));


    // Inputs: input_ids, attention_mask, position_ids, ...
    // Outputs: logits, ...
    const auto kStartInputKVCacheLayers = 3u;
    const auto kStartOutputKVCacheLayers = 1u;

    const auto& kvcache_compiled = m_kvcache_request.get_compiled_model();
    for (int i = 0; i < kvcache_compiled.outputs().size() - 1; ++i) {
        const auto& input_name = kvcache_compiled.inputs()[kStartInputKVCacheLayers + i].get_any_name();
        const auto& output_name = kvcache_compiled.outputs()[kStartOutputKVCacheLayers + i].get_any_name();
        auto kvcache_out_tensor = m_kvcache_request.get_tensor(output_name);
        m_kvcache_request.set_tensor(input_name, kvcache_out_tensor);
        auto prefill_tensor = m_prefill_request.get_tensor(output_name);
        auto kvcache_tensor = m_kvcache_request.get_tensor(input_name);
        prefill_tensor.copy_to(kvcache_tensor);
    }

    auto* input_ids_data = m_kvcache_request.get_tensor("input_ids").data<int64_t>();
    auto* position_ids_data = m_kvcache_request.get_tensor("position_ids").data<int64_t>();
    auto* attention_mask_data = m_kvcache_request.get_tensor("attention_mask").data<int64_t>();

    const size_t max_tokens = config.get_max_new_tokens(prompt_len);
    for (int i = 0; i < max_tokens - 1; ++i) {
        input_ids_data[0] = last_token;
        position_ids_data[0] = m_kvcache_desc.num_stored_tokens;
        attention_mask_data[m_kvcache_desc.total_size - m_kvcache_desc.num_stored_tokens - 1] = 1u;

        m_kvcache_request.infer();
        m_kvcache_desc.num_stored_tokens += 1;

        last_token = utils::argmax(m_kvcache_request.get_tensor("logits"), 0);
        results.tokens[0].push_back(last_token);
        results.scores[0] = 0u;

        if (streamer_ptr && streamer_ptr->put(last_token)) {
            break;
        }

        if (last_token == m_generation_config.eos_token_id) {
            break;
        }

        // NB: KV-cache is full, further generation is impossible
        if (m_kvcache_desc.num_stored_tokens == m_kvcache_desc.total_size) {
            break;
        }

    }
    return results;
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
    if (device == "NPU") {
        m_pimpl = make_unique<NPULLMPipelineImpl>(std::filesystem::path(model_path), tokenizer, plugin_config);
    } else {
        m_pimpl = make_unique<LLMPipelineImpl>(std::filesystem::path(model_path), tokenizer, device, plugin_config);
    }
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::string& path,
    const std::string& device,
    const ov::AnyMap& config
) {
    if (device == "NPU") {
        m_pimpl = make_unique<NPULLMPipelineImpl>(std::filesystem::path(path), config);
    } else {
        m_pimpl = make_unique<LLMPipelineImpl>(std::filesystem::path(path), device, config);
    }
}

ov::genai::GenerationConfig ov::genai::LLMPipeline::get_generation_config() const {
    return m_pimpl->m_generation_config;
}

ov::genai::Tokenizer ov::genai::LLMPipeline::get_tokenizer() {
    return m_pimpl->m_tokenizer;
}

void ov::genai::LLMPipeline::start_chat() {
    m_pimpl->start_chat();
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
