// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include <nlohmann/json.hpp>

#include "openvino/core/visibility.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/perf_metrics.hpp"

#include "llm/pipeline_static.hpp"
#include "llm/pipeline_stateful.hpp"
#include "llm/pipeline_continuous_batching_adapter.hpp"
#include "speculative_decoding/speculative_decoding_impl.hpp"
#include "utils.hpp"

namespace ov {

namespace genai {

std::pair<std::string, Any> streamer(StreamerVariant func) {
    if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&func)) {
        return {utils::STREAMER_ARG_NAME, Any::make<std::shared_ptr<StreamerBase>>(*streamer_obj)};
    } else if (auto streamer_obj = std::get_if<std::function<StreamingStatus(std::string)>>(&func)) {
        return {utils::STREAMER_ARG_NAME, Any::make<std::function<StreamingStatus(std::string)>>(*streamer_obj)};
    } else {
        auto callback = std::get<std::function<bool(std::string)>>(func);
        return {utils::STREAMER_ARG_NAME, Any::make<std::function<bool(std::string)>>(callback)};
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
    auto generation_config = utils::from_config_json_if_exists(models_path);
    auto tokenizer = ov::genai::Tokenizer(models_path);
    return { utils::DRAFT_MODEL_ARG_NAME, Any::make<ModelDesc>(model, tokenizer, device, plugin_config, scheduler_config, generation_config) };
}

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
    auto [properties, attention_backend] = utils::extract_attention_backend(user_properties);

    // If CB is invoked explicitly, create CB adapter as is and re-throw in case if internal issues
    if (utils::explicitly_requires_paged_attention(user_properties)) {
        auto [device_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
        m_pimpl = std::make_unique<ContinuousBatchingAdapter>(models_path, tokenizer, scheduler_config, device, device_properties);
    } else if (device == "NPU") {
        m_pimpl = properties.count("STATIC_PIPELINE")
            ? static_llm::LLMPipelineFactory::create(models_path, tokenizer, properties)
            : std::make_unique<StatefulLLMPipeline>(models_path, tokenizer, device, properties);
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

    auto [properties, attention_backend] = utils::extract_attention_backend(user_properties);

    // If CB is invoked explicitly, create CB adapter as is and re-throw in case if internal issues
    if (utils::explicitly_requires_paged_attention(user_properties)) {
        auto [device_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
        m_pimpl = std::make_unique<ContinuousBatchingAdapter>(models_path, scheduler_config, device, device_properties);
    } else if (device == "NPU") {
        m_pimpl = properties.count("STATIC_PIPELINE")
            ? static_llm::LLMPipelineFactory::create(models_path, properties)
            : std::make_unique<StatefulLLMPipeline>(models_path, device, properties);
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

    auto [properties, attention_backend] = utils::extract_attention_backend(user_properties);

    // If CB is invoked explicitly, create CB adapter as is and re-throw in case if internal issues
    if (utils::explicitly_requires_paged_attention(user_properties)) {
        auto [device_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
        m_pimpl = std::make_unique<ContinuousBatchingAdapter>(model_str, weights_tensor,
                                                              tokenizer, scheduler_config, device, device_properties, generation_config);
    } else if (device == "NPU") {
        m_pimpl = properties.count("STATIC_PIPELINE")
            ? static_llm::LLMPipelineFactory::create(
                  utils::singleton_core().read_model(model_str, weights_tensor),
                  tokenizer,
                  properties,
                  generation_config)
            : std::make_unique<StatefulLLMPipeline>(
                utils::singleton_core().read_model(model_str, weights_tensor),
                tokenizer,
                device,
                properties,
                generation_config);
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
    StreamerVariant streamer) {
    return m_pimpl->generate(inputs, generation_config, streamer);
}

EncodedResults LLMPipeline::generate(const EncodedInputs& inputs, const ov::AnyMap& config_map) {
    auto config_arg = utils::get_config_from_map(config_map);
    GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
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

ov::genai::LLMPipeline::~LLMPipeline() {
    m_pimpl.reset();
    utils::release_core_plugin(m_device);
}

} // namespace genai
} // namespace ov
