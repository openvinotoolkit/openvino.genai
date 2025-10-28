// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mutex>
#include <memory>
#include <openvino/runtime/properties.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "speculative_decoding/speculative_decoding_impl.hpp"
#include "prompt_lookup/prompt_lookup_impl.hpp"
#include "continuous_batching/timer.hpp"
#include "utils.hpp"
#include "visual_language/inputs_embedder.hpp"

using namespace ov::genai;

namespace {
bool
extract_prompt_lookup_from_config(ov::AnyMap& config) {
    bool res = false;
    if (config.find(ov::genai::prompt_lookup.name()) != config.end()) {
        res = config.at(ov::genai::prompt_lookup.name()).as<bool>();
        config.erase(ov::genai::prompt_lookup.name());
    }
    return res;
}

float get_load_time(std::chrono::steady_clock::time_point start_time) {
    auto stop_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline( const std::filesystem::path& models_path,
                                                        const SchedulerConfig& scheduler_config,
                                                        const std::string& device,
                                                        const ov::AnyMap& properties,
                                                        const ov::AnyMap& tokenizer_properties,
                                                        const ov::AnyMap& vision_encoder_properties) {
    auto start_time = std::chrono::steady_clock::now();
    auto properties_without_draft_model = properties;
    auto draft_model_desr = utils::extract_draft_model_from_config(properties_without_draft_model);
    auto is_prompt_lookup_enabled = extract_prompt_lookup_from_config(properties_without_draft_model);

    auto model = utils::read_model(models_path, properties);
    auto [properties_without_draft_model_without_gguf, enable_save_ov_model] = utils::extract_gguf_properties(properties_without_draft_model);
    properties_without_draft_model_without_gguf[ov::cache_model_path.name()] = models_path;
    auto tokenizer = ov::genai::Tokenizer(models_path, tokenizer_properties);
    auto generation_config = utils::from_config_json_if_exists(models_path);

    std::shared_ptr<InputsEmbedder> embedder;
    if (std::filesystem::exists(models_path / "openvino_text_embeddings_model.xml")) {
        embedder = std::make_shared<InputsEmbedder>(models_path, device, vision_encoder_properties);
    }

    utils::print_scheduler_config_info(scheduler_config);

    if (is_prompt_lookup_enabled) {
        OPENVINO_ASSERT(draft_model_desr.model == nullptr, "Speculative decoding and prompt lookup decoding are mutually exclusive");
        OPENVINO_ASSERT(embedder == nullptr, "Prompt lookup decoding is not supported for models with embeddings");
        m_impl = std::make_shared<PromptLookupImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model_without_gguf, generation_config);
    } else if (draft_model_desr.model != nullptr) {
        OPENVINO_ASSERT(embedder == nullptr, "Speculative decoding is not supported for models with embeddings");
        auto main_model_descr = ov::genai::ModelDesc(model, tokenizer, device, properties_without_draft_model_without_gguf, scheduler_config, generation_config);
        m_impl = std::make_shared<SpeculativeDecodingImpl>(main_model_descr, draft_model_desr);
    } else if (embedder) {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, embedder, tokenizer, scheduler_config, device, properties_without_draft_model_without_gguf, generation_config);
    }
    else {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model_without_gguf, generation_config);
    }

    m_impl->m_load_time_ms = get_load_time(start_time);
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline(
    const std::filesystem::path& models_path,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    auto properties_without_draft_model = properties;
    auto draft_model_desr = utils::extract_draft_model_from_config(properties_without_draft_model);
    auto is_prompt_lookup_enabled = extract_prompt_lookup_from_config(properties_without_draft_model);

    auto model = utils::read_model(models_path, properties_without_draft_model);
    auto [properties_without_draft_model_without_gguf, enable_save_ov_model] = utils::extract_gguf_properties(properties_without_draft_model);
    properties_without_draft_model_without_gguf[ov::cache_model_path.name()] = models_path;

    auto generation_config = utils::from_config_json_if_exists(models_path);

    std::shared_ptr<InputsEmbedder> embedder;
    if (std::filesystem::exists(models_path / "openvino_text_embeddings_model.xml")) {
        embedder = std::make_shared<InputsEmbedder>(models_path, device, properties_without_draft_model_without_gguf);
    }

    utils::print_scheduler_config_info(scheduler_config);

    if (is_prompt_lookup_enabled) {
        OPENVINO_ASSERT(draft_model_desr.model == nullptr, "Speculative decoding and prompt lookup decoding are mutually exclusive");
        OPENVINO_ASSERT(embedder == nullptr, "Prompt lookup decoding is not supported for models with embeddings");
        m_impl = std::make_shared<PromptLookupImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model_without_gguf, generation_config);
    } else if (draft_model_desr.model != nullptr) {
        OPENVINO_ASSERT(embedder == nullptr, "Speculative decoding is not supported for models with embeddings");
        auto main_model_descr = ov::genai::ModelDesc(model, tokenizer, device, properties_without_draft_model_without_gguf, scheduler_config, generation_config);
        m_impl = std::make_shared<SpeculativeDecodingImpl>(main_model_descr, draft_model_desr);
    } else if (embedder) {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, embedder, tokenizer, scheduler_config, device, properties_without_draft_model_without_gguf, generation_config);
    } else {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model_without_gguf, generation_config);
    }

    m_impl->m_load_time_ms = get_load_time(start_time);
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline(
    const std::string& model_str,
    const ov::Tensor& weights_tensor,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config) {
    auto start_time = std::chrono::steady_clock::now();

    auto properties_without_draft_model = properties;
    auto draft_model_desr = utils::extract_draft_model_from_config(properties_without_draft_model);
    auto is_prompt_lookup_enabled = extract_prompt_lookup_from_config(properties_without_draft_model);
    auto model = utils::singleton_core().read_model(model_str, weights_tensor);

    auto rt_info = model->get_rt_info();
    std::shared_ptr<InputsEmbedder> embedder = nullptr;
    std::filesystem::path directory;
    if (rt_info.find("__weights_path") != rt_info.end()) {
        std::string weights_path = rt_info.at("__weights_path").as<std::string>();
        directory = std::filesystem::path(weights_path).parent_path();
        if (std::filesystem::exists(directory / "openvino_text_embeddings_model.xml")) {
            embedder = std::make_shared<InputsEmbedder>(directory, device, properties_without_draft_model);
        }
    }

    utils::print_scheduler_config_info(scheduler_config);

    if (is_prompt_lookup_enabled) {
        OPENVINO_ASSERT(draft_model_desr.model == nullptr, "Speculative decoding and prompt lookup decoding are mutually exclusive");
        OPENVINO_ASSERT(embedder == nullptr, "Prompt lookup decoding is not supported for models with embeddings");
        m_impl = std::make_shared<PromptLookupImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model, generation_config);
    } else if (draft_model_desr.model != nullptr) {
        OPENVINO_ASSERT(embedder == nullptr, "Speculative decoding is not supported for models with embeddings");
        auto main_model_descr = ov::genai::ModelDesc(model, tokenizer, device, properties_without_draft_model, scheduler_config, generation_config);
        m_impl = std::make_shared<SpeculativeDecodingImpl>(main_model_descr, draft_model_desr);
    } else if (embedder) {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, embedder, tokenizer, scheduler_config, device, properties_without_draft_model, generation_config);
    } else {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model, generation_config);
    }

    m_impl->m_load_time_ms = get_load_time(start_time);
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline(
        const ModelsMap& models_map,
        const ov::genai::Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        std::optional<std::filesystem::path> embedder_config_dir_path,
        const ov::AnyMap& properties,
        const ov::genai::GenerationConfig& generation_config) {
    auto start_time = std::chrono::steady_clock::now();

    auto properties_without_draft_model = properties;
    auto draft_model_desr = utils::extract_draft_model_from_config(properties_without_draft_model);
    auto is_prompt_lookup_enabled = extract_prompt_lookup_from_config(properties_without_draft_model);
    auto model_pair = utils::get_model_weights_pair(models_map, "language");
    auto model = utils::singleton_core().read_model(model_pair.first, model_pair.second);

    auto rt_info = model->get_rt_info();
    std::filesystem::path directory;
    std::shared_ptr<InputsEmbedder> embedder = nullptr;
    if (embedder_config_dir_path.has_value()) {
        auto path = *embedder_config_dir_path;
        embedder = std::make_shared<InputsEmbedder>(models_map, tokenizer, path, device, properties);
    }
    else if (rt_info.find("__weights_path") != rt_info.end()) {
        std::string weights_path = rt_info.at("__weights_path").as<std::string>();
        directory = std::filesystem::path(weights_path).parent_path();
        if (std::filesystem::exists(directory / "openvino_text_embeddings_model.xml")) {
            embedder = std::make_shared<InputsEmbedder>(directory, device, properties_without_draft_model);
        }
    }

    utils::print_scheduler_config_info(scheduler_config);

    if (is_prompt_lookup_enabled) {
        OPENVINO_ASSERT(draft_model_desr.model == nullptr, "Speculative decoding and prompt lookup decoding are mutually exclusive");
        OPENVINO_ASSERT(embedder == nullptr, "Prompt lookup decoding is not supported for models with embeddings");
        m_impl = std::make_shared<PromptLookupImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model, generation_config);
    } else if (draft_model_desr.model != nullptr) {
        OPENVINO_ASSERT(embedder == nullptr, "Speculative decoding is not supported for models with embeddings");
        auto main_model_descr = ov::genai::ModelDesc(model, tokenizer, device, properties_without_draft_model, scheduler_config, generation_config);
        m_impl = std::make_shared<SpeculativeDecodingImpl>(main_model_descr, draft_model_desr);
    } else if (embedder) {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, embedder, tokenizer, scheduler_config, device, properties_without_draft_model, generation_config);
    } else {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model, generation_config);
    }

    m_impl->m_load_time_ms = get_load_time(start_time);
}

ov::genai::Tokenizer ContinuousBatchingPipeline::get_tokenizer() const{
    return m_impl->get_tokenizer();
}

ov::genai::GenerationConfig ContinuousBatchingPipeline::get_config() const{
    return m_impl->get_config();
}

void ContinuousBatchingPipeline::set_config(const ov::genai::GenerationConfig& config) {
    m_impl->set_config(config);
}

PipelineMetrics ContinuousBatchingPipeline::get_metrics() const{
    return m_impl->get_metrics();
}

GenerationHandle ContinuousBatchingPipeline::add_request(uint64_t request_id, const std::string& prompt, const ov::genai::GenerationConfig& sampling_params) {
    return m_impl->add_request(request_id, prompt, sampling_params);
}

GenerationHandle ContinuousBatchingPipeline::add_request(uint64_t request_id, const ov::Tensor& input_ids, const ov::genai::GenerationConfig& sampling_params) {
    return m_impl->add_request(request_id, input_ids, sampling_params);
}

GenerationHandle ContinuousBatchingPipeline::add_request(uint64_t request_id, const std::string& prompt, const std::vector<ov::Tensor>& images, const ov::genai::GenerationConfig& sampling_params) {
    return m_impl->add_request(request_id, prompt, images, sampling_params);
}

GenerationHandle ContinuousBatchingPipeline::add_request(uint64_t request_id, const std::string& prompt, const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos, const ov::genai::GenerationConfig& sampling_params) {
    return m_impl->add_request(request_id, prompt, images, videos, sampling_params);
}

void ContinuousBatchingPipeline::step() {
    m_impl->step();
}

bool ContinuousBatchingPipeline::has_non_finished_requests() {
    return m_impl->has_non_finished_requests();
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::generate(const std::vector<ov::Tensor>& input_ids, const std::vector<ov::genai::GenerationConfig>& sampling_params, const StreamerVariant& streamer) {
    auto encoded_results = m_impl->generate(input_ids, sampling_params, streamer);

    for (auto& encoded_result : encoded_results) {
        encoded_result.perf_metrics.load_time = m_impl->m_load_time_ms;
    }

    return encoded_results;
}

std::vector<GenerationResult> ContinuousBatchingPipeline::generate(const std::vector<std::string>& prompts, const std::vector<ov::genai::GenerationConfig>& sampling_params, const StreamerVariant& streamer) {
    auto decoded_results = m_impl->generate(prompts, sampling_params, streamer);

    for (auto& decoded_result : decoded_results) {
        decoded_result.perf_metrics.load_time = m_impl->m_load_time_ms;
    }

    return decoded_results;
}

std::vector<GenerationResult> ContinuousBatchingPipeline::generate(
    const std::vector<ChatHistory>& histories,
    const std::vector<ov::genai::GenerationConfig>&
    sampling_params,
    const StreamerVariant& streamer
) {
    auto decoded_results = m_impl->generate(histories, sampling_params, streamer);

    for (auto& decoded_result : decoded_results) {
        decoded_result.perf_metrics.load_time = m_impl->m_load_time_ms;
    }

    return decoded_results;
}

std::vector<VLMDecodedResults> ContinuousBatchingPipeline::generate(
             const std::vector<std::string>& prompts,
             const std::vector<std::vector<ov::Tensor>>& images,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer) {
    return m_impl->generate(prompts, images, sampling_params, streamer);
}

std::vector<VLMDecodedResults> ContinuousBatchingPipeline::generate(
    const std::vector<std::string>& prompts,
    const std::vector<std::vector<ov::Tensor>>& images,
    const std::vector<std::vector<ov::Tensor>>& video,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer) {
    return m_impl->generate(prompts, images, video, sampling_params, streamer);
}


void ContinuousBatchingPipeline::start_chat(const std::string& system_message) {
    m_impl->finish_chat();
    m_impl->start_chat(system_message);
}

void ContinuousBatchingPipeline::finish_chat() {
    m_impl->finish_chat();
}
