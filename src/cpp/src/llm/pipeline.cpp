// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include <nlohmann/json.hpp>

#include "openvino/core/visibility.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/text_streamer.hpp"

#include "llm/pipeline_stateful.hpp"
#include "llm/pipeline_continuous_batching_adapter.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "speculative_decoding/stateful/eagle3_strategy.hpp"
#include "speculative_decoding/stateful/fast_draft_strategy.hpp"
#include "utils.hpp"

namespace {

// This is a decorator function that wraps a generation callable to apply parsers and reset them before generation if needed.
ov::genai::DecodedResults run_generate_with_parsers(const ov::genai::OptionalGenerationConfig& generation_config,
                 const ov::genai::StreamerVariant& streamer,
                std::function<ov::genai::DecodedResults(void)> generate_callable) {
                    
    std::shared_ptr<ov::genai::TextParserStreamer> parser_streamer;
    // If streamer is of StreamerBase type, and it is TextParserStreamer, get parsed message
    // Streaming is available only for batch size 1 therefore only parsed[0]
    if (auto streamer_obj = std::get_if<std::shared_ptr<ov::genai::StreamerBase>>(&streamer)) {
        parser_streamer = std::dynamic_pointer_cast<ov::genai::TextParserStreamer>(*streamer_obj);
    }

    // determine from generation config when 'need_to_reset_parser' will be available
    // TODO: Determine 'need_to_reset_parser' from generation_config when available.
    bool need_to_reset_parser = true;
    if (parser_streamer && need_to_reset_parser) {
        parser_streamer->reset();
    }

    auto res = generate_callable();
    
    if (parser_streamer) {
        res.parsed.resize(1);
        res.parsed[0] = parser_streamer->get_parsed_message();
    }

    // If no parsers are defined, return
    if (!generation_config.has_value() || generation_config->parsers.empty()) {
        return res;
    }
    
    std::vector<std::shared_ptr<ov::genai::Parser>> parsers = generation_config->parsers;
    res.parsed.resize(res.texts.size());
    
    // Apply Base parsers sequentially even if IncrementalParser has run.
    for (size_t i = 0; i < res.texts.size(); ++i) {
        auto& msg = res.parsed[i];
        if (!msg.contains("content")) {
            // Initialize msg with content
            msg["content"] = res.texts[i];
        }
        
        for (auto& parser: parsers) {
            parser->parse(msg);
        }
        res.parsed[i] = msg;
    }
    return res;
}

}

namespace ov {

namespace genai {

std::pair<std::string, Any> streamer(StreamerVariant func) {
    if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&func)) {
        return {utils::STREAMER_ARG_NAME, Any::make<std::shared_ptr<StreamerBase>>(*streamer_obj)};
    } else {
        auto status_streamer_obj = std::get<std::function<StreamingStatus(std::string)>>(func);
        return {utils::STREAMER_ARG_NAME, Any::make<std::function<StreamingStatus(std::string)>>(status_streamer_obj)};
    }
}

std::pair<std::string, Any> generation_config(const GenerationConfig& config) {
    return {utils::CONFIG_ARG_NAME, Any::make<GenerationConfig>(config)};
}

std::pair<std::string, Any> draft_model(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& properties) {
    auto [plugin_config, scheduler_config] = utils::extract_scheduler_config(properties);

    std::filesystem::path openvino_model_name = "openvino_model.xml";
    auto model = utils::singleton_core().read_model(models_path / openvino_model_name, {}, plugin_config);
    utils::eagle3::apply_eagle3_rt_info(model, plugin_config);
    auto generation_config = utils::from_config_json_if_exists(models_path);
    auto tokenizer = ov::genai::Tokenizer(models_path);
    return { utils::DRAFT_MODEL_ARG_NAME, Any::make<ModelDesc>(model, tokenizer, device, plugin_config, scheduler_config, generation_config) };
}

std::pair<std::string, Any> draft_model(
    std::string& model_str,
    ov::Tensor& weights_tensor,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config) {
    auto [plugin_config, scheduler_config] = utils::extract_scheduler_config(properties);

    auto model = utils::singleton_core().read_model(model_str, weights_tensor);
    utils::eagle3::apply_eagle3_rt_info(model, plugin_config);
    return { utils::DRAFT_MODEL_ARG_NAME, Any::make<ModelDesc>(model, tokenizer, device, plugin_config, scheduler_config, generation_config) };
}

class StatefulPipeline {
public:
static std::unique_ptr<LLMPipelineImplBase> create(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties) {
    return create(ov::genai::utils::read_model(models_path, properties),
                  tokenizer,
                  device,
                  properties,
                  utils::from_config_json_if_exists(models_path),
                  models_path);
}

static std::unique_ptr<LLMPipelineImplBase> create(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& plugin_config) {
    return create(models_path, Tokenizer(models_path, plugin_config), device, plugin_config);
}

static std::unique_ptr<LLMPipelineImplBase> create(const std::shared_ptr<ov::Model>& model,
                                                   const ov::genai::Tokenizer& tokenizer,
                                                   const std::string& device,
                                                   const ov::AnyMap& properties,
                                                   const ov::genai::GenerationConfig& generation_config,
                                                   const std::filesystem::path& models_path = {}) {
    auto properties_without_draft_model = properties;
    auto draft_model_descr = ov::genai::utils::extract_draft_model_from_config(properties_without_draft_model);

    auto main_model_descr =
        ov::genai::ModelDesc(model, tokenizer, device, properties_without_draft_model, {}, generation_config);

    if (draft_model_descr.model != nullptr) {
        // FIXME: Add support for StatefulSpeculativeLLMPipeline for non-NPU devices for both models.
        OPENVINO_ASSERT(device == "NPU" || draft_model_descr.device == "NPU",
                        "Stateful FastDraft and Stateful Eagle3 Speculative Decoding require NPU to be "
                        "the execution device for at least one model.");

        // Check if Eagle3 mode is enabled in draft model properties
        bool is_eagle3_mode = draft_model_descr.properties.find("eagle3_mode") != draft_model_descr.properties.end() &&
                              draft_model_descr.properties.at("eagle3_mode").as<bool>();

        if (is_eagle3_mode) {
            // Eagle3 Speculative Decoding mode
            auto eagle_rt_info = utils::eagle3::extract_eagle3_info_from_config(draft_model_descr.properties, models_path);
            if (!eagle_rt_info.hidden_layers_list.empty()) {
                draft_model_descr.properties["hidden_layers_list"] = eagle_rt_info.hidden_layers_list;
            }
            return std::make_unique<StatefulEagle3LLMPipeline>(main_model_descr, draft_model_descr);
        } else {
            // Standard Speculative Decoding mode (FastDraft)
            return std::make_unique<StatefulSpeculativeLLMPipeline>(main_model_descr, draft_model_descr);
        }
    }

    return std::make_unique<StatefulLLMPipeline>(main_model_descr.model,
                                                 main_model_descr.tokenizer,
                                                 main_model_descr.device,
                                                 main_model_descr.properties,
                                                 main_model_descr.generation_config);
}
};

// Public LLMPipeline

ov::genai::LLMPipeline::LLMPipeline(
    const ov::InferRequest& request,
    const ov::genai::Tokenizer& tokenizer,
    OptionalGenerationConfig generation_config) {
    auto start_time = std::chrono::steady_clock::now();
    m_pimpl = std::make_unique<StatefulLLMPipeline>(request, tokenizer, generation_config);
    m_pimpl->save_load_time(start_time);
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& user_properties) :
    m_device(device) {
    auto start_time = std::chrono::steady_clock::now();

    bool is_npu_requested = ov::genai::utils::is_npu_requested(device, user_properties);
    auto [properties, attention_backend] = utils::extract_attention_backend(user_properties, is_npu_requested);

    if (is_npu_requested) {
        m_pimpl = StatefulPipeline::create(models_path, tokenizer, device, properties);
    } else if (utils::explicitly_requires_paged_attention(user_properties)) {
        // If CB is invoked explicitly, create CB adapter as is and re-throw in case if internal issues
        auto [device_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
        m_pimpl = std::make_unique<ContinuousBatchingAdapter>(models_path, tokenizer, scheduler_config, device, device_properties);
    } else if (attention_backend == PA_BACKEND) {
        // try to call CB adapter one more time, but with safe guard to silent exception
        try {
            // we need use CB only for x86 and arm64, as for other architectures like risc-v we can create Paged Attention based model
            // but cannot perform its inference later
#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
            m_pimpl = std::make_unique<ContinuousBatchingAdapter>(models_path, tokenizer, utils::get_latency_oriented_scheduler_config(), device, properties);
#endif
        } catch (ov::Exception&) {
            // ignore exceptions from PA
        }
    }

    if (m_pimpl == nullptr) {
        // FIXME: Switch to StatefulPipeline::create after resolving issues
        //        with GPU and CPU for StatefulSpeculativeLLMPipeline
        m_pimpl = std::make_unique<StatefulLLMPipeline>(models_path, tokenizer, device, properties);
    }

    m_pimpl->save_load_time(start_time);
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& user_properties) :
    m_device(device) {
    auto start_time = std::chrono::steady_clock::now();

    bool is_npu_requested = ov::genai::utils::is_npu_requested(device, user_properties);
    auto [properties, attention_backend] = utils::extract_attention_backend(user_properties, is_npu_requested);

    if (is_npu_requested) {
        m_pimpl = StatefulPipeline::create(models_path, device, properties);
    } else if (utils::explicitly_requires_paged_attention(user_properties)) {
        // If CB is invoked explicitly, create CB adapter as is and re-throw in case if internal issues
        auto [device_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
        m_pimpl = std::make_unique<ContinuousBatchingAdapter>(models_path, scheduler_config, device, device_properties);
    } else if (attention_backend == PA_BACKEND) {
        // try to call CB adapter one more time, but with safe guard to silent exception
        try {
            // we need use CB only for x86 and arm64, as for other architectures like risc-v we can create Paged Attention based model
            // but cannot perform its inference later
#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
            m_pimpl = std::make_unique<ContinuousBatchingAdapter>(models_path, utils::get_latency_oriented_scheduler_config(), device, properties);
#endif
        } catch (ov::Exception&) {
            // ignore exceptions from PA
        }
    }

    if (m_pimpl == nullptr) {
        // FIXME: Switch to StatefulPipeline::create after resolving issues
        //        with GPU and CPU for StatefulSpeculativeLLMPipeline
        m_pimpl = std::make_unique<StatefulLLMPipeline>(models_path, device, properties);
    }

    m_pimpl->save_load_time(start_time);
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::string& model_str,
    const ov::Tensor& weights_tensor,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& user_properties,
    const ov::genai::GenerationConfig& generation_config) :
    m_device(device) {
    auto start_time = std::chrono::steady_clock::now();

    bool is_npu_requested = ov::genai::utils::is_npu_requested(device, user_properties);
    auto [properties, attention_backend] = utils::extract_attention_backend(user_properties, is_npu_requested);

    if (is_npu_requested) {
        m_pimpl = StatefulPipeline::create(
            utils::singleton_core().read_model(model_str, weights_tensor),
            tokenizer,
            device,
            properties,
            generation_config);
    } else if (utils::explicitly_requires_paged_attention(user_properties)) {
        // If CB is invoked explicitly, create CB adapter as is and re-throw in case if internal issues
        auto [device_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
        m_pimpl = std::make_unique<ContinuousBatchingAdapter>(model_str, weights_tensor,
                                                              tokenizer, scheduler_config, device, device_properties, generation_config);
    } else if (attention_backend == PA_BACKEND) {
        // try to call CB adapter one more time, but with safe guard to silent exception
        try {
            // we need use CB only for x86 and arm64, as for other architectures like risc-v we can create Paged Attention based model
            // but cannot perform its inference later
#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
            m_pimpl = std::make_unique<ContinuousBatchingAdapter>(model_str, weights_tensor, tokenizer,
                                                                  utils::get_latency_oriented_scheduler_config(), device, properties, generation_config);
#endif
        } catch (ov::Exception&) {
            // ignore exceptions from PA
        }
    }

    if (m_pimpl == nullptr) {
        // FIXME: Switch to StatefulPipeline::create after resolving issues
        //        with GPU and CPU for StatefulSpeculativeLLMPipeline
        m_pimpl = std::make_unique<StatefulLLMPipeline>(
            utils::singleton_core().read_model(model_str, weights_tensor),
            tokenizer,
            device,
            properties,
            generation_config);
    }

    m_pimpl->save_load_time(start_time);
}

DecodedResults LLMPipeline::generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer) {

    return run_generate_with_parsers(generation_config, streamer, [&]() -> DecodedResults {
        return m_pimpl->generate(inputs, generation_config, streamer);
    });
}

DecodedResults LLMPipeline::generate(StringInputs text, const ov::AnyMap& config_map) {
    auto config_arg = utils::get_config_from_map(config_map);
    GenerationConfig config = config_arg.value_or(get_generation_config());
    config.update_generation_config(config_map);
    auto streamer = utils::get_streamer_from_map(config_map);
    
    return run_generate_with_parsers(config_arg, streamer, [&]() -> DecodedResults {
        return m_pimpl->generate(text, config, streamer);
    });
}

DecodedResults LLMPipeline::generate(
        const ChatHistory& history,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer) {
    return run_generate_with_parsers(generation_config, streamer, [&]() -> DecodedResults {
        return m_pimpl->generate(history, generation_config, streamer);
    });
}

DecodedResults LLMPipeline::generate(const ChatHistory& history, const ov::AnyMap& config_map) {
    auto config_arg = utils::get_config_from_map(config_map);
    GenerationConfig config = config_arg.value_or(get_generation_config());
    config.update_generation_config(config_map);
    auto streamer = utils::get_streamer_from_map(config_map);

    return run_generate_with_parsers(config, streamer, [&]() -> DecodedResults {
        return m_pimpl->generate(history, config, streamer);
    });
}

EncodedResults LLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
    return m_pimpl->generate(inputs, generation_config, streamer);
}

EncodedResults LLMPipeline::generate(const EncodedInputs& inputs, const ov::AnyMap& config_map) {
    auto config_arg = utils::get_config_from_map(config_map);
    GenerationConfig config = config_arg.value_or(get_generation_config());
    config.update_generation_config(config_map);

    return m_pimpl->generate(inputs, config, utils::get_streamer_from_map(config_map));
}

ov::genai::GenerationConfig ov::genai::LLMPipeline::get_generation_config() const {
    return m_pimpl->get_generation_config();
}

ov::genai::Tokenizer ov::genai::LLMPipeline::get_tokenizer() {
    return m_pimpl->get_tokenizer();
}

void ov::genai::LLMPipeline::start_chat(const std::string& system_message) {
    m_pimpl->start_chat(system_message);
}

void ov::genai::LLMPipeline::finish_chat() {
    m_pimpl->finish_chat();
}

void ov::genai::LLMPipeline::set_generation_config(const GenerationConfig& config) {
    m_pimpl->set_generation_config(config);
}

std::vector<float> ov::genai::LLMPipeline::get_next_token_log_probs(
    const std::string& prompt,
    const std::vector<int64_t>& token_ids
) {
    return m_pimpl->get_next_token_log_probs(prompt, token_ids);
}

ov::genai::LLMPipeline::~LLMPipeline() {
    m_pimpl.reset();
    utils::release_core_plugin(m_device);
}

} // namespace genai
} // namespace ov
