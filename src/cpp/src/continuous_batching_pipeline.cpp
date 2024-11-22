// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mutex>
#include <memory>
#include <openvino/runtime/properties.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "continuous_batching_impl.hpp"
#include "speculative_decoding/speculative_decoding_impl.hpp"
#include "prompt_lookup/prompt_lookup_impl.hpp"
#include "timer.hpp"
#include "utils.hpp"
#include "debug_utils.hpp"
#include "cache_state_dumper.hpp"

using namespace ov::genai;

inline ov::genai::ModelDesc
extract_draft_model_from_config(ov::AnyMap& config) {
    ov::genai::ModelDesc draft_model("");
    if (config.find(utils::DRAFT_MODEL_ARG_NAME) != config.end()) {
        draft_model = config.at(utils::DRAFT_MODEL_ARG_NAME).as<ov::genai::ModelDesc>();
        config.erase(utils::DRAFT_MODEL_ARG_NAME);
    }
    return draft_model;
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline( const std::filesystem::path& models_path,
                                                        const SchedulerConfig& scheduler_config,
                                                        const std::string& device,
                                                        const ov::AnyMap& properties,
                                                        const ov::AnyMap& tokenizer_properties) {
    auto properties_without_draft_model = properties;
    auto draft_model = extract_draft_model_from_config(properties_without_draft_model);
    if (properties_without_draft_model.count(ov::genai::enable_prompt_lookup.name())) {
        properties_without_draft_model.erase(ov::genai::enable_prompt_lookup.name());
        m_impl = std::make_shared<PromptLookupImpl>(models_path, scheduler_config, device, properties_without_draft_model, tokenizer_properties);
    } else if (draft_model.models_path.empty()) {
        m_impl = std::make_shared<ContinuousBatchingImpl>(models_path, scheduler_config, device, properties, tokenizer_properties);
    } else {
        m_impl = std::make_shared<SpeculativeDecodingImpl>(models_path, scheduler_config, device, properties_without_draft_model, draft_model, tokenizer_properties);
    }
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline(
    const std::filesystem::path& models_path,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties) {
    auto properties_without_draft_model = properties;
    auto draft_model = extract_draft_model_from_config(properties_without_draft_model);
    if (properties_without_draft_model.count(ov::genai::enable_prompt_lookup.name())) {
        properties_without_draft_model.erase(ov::genai::enable_prompt_lookup.name());
        m_impl = std::make_shared<PromptLookupImpl>(models_path, tokenizer, scheduler_config, device, properties_without_draft_model);
    } else if (draft_model.models_path.empty()) {
        m_impl = std::make_shared<ContinuousBatchingImpl>(models_path, tokenizer, scheduler_config, device, properties);
    } else {
        m_impl = std::make_shared<SpeculativeDecodingImpl>(models_path, scheduler_config, device, properties_without_draft_model, draft_model);
    }
}

ov::genai::Tokenizer ContinuousBatchingPipeline::get_tokenizer() {
    return m_impl->get_tokenizer();
}

ov::genai::GenerationConfig ContinuousBatchingPipeline::get_config() const{
    return m_impl->get_config();
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

void ContinuousBatchingPipeline::step() {
    m_impl->step();
}

bool ContinuousBatchingPipeline::has_non_finished_requests() {
    return m_impl->has_non_finished_requests();
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::generate(const std::vector<ov::Tensor>& input_ids, const std::vector<ov::genai::GenerationConfig>& sampling_params, const StreamerVariant& streamer) {
    return m_impl->generate(input_ids, sampling_params, streamer);
}

std::vector<GenerationResult> ContinuousBatchingPipeline::generate(const std::vector<std::string>& prompts, const std::vector<ov::genai::GenerationConfig>& sampling_params, const StreamerVariant& streamer) {
    return m_impl->generate(prompts, sampling_params, streamer);
}

void ContinuousBatchingPipeline::start_chat(const std::string& system_message) {
    m_impl->start_chat(system_message);
};

void ContinuousBatchingPipeline::finish_chat() {
    m_impl->finish_chat();
};
