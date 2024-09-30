// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_impl.hpp"

namespace ov::genai {
ContinuousBatchingPipeline::SpeculativeDecodingImpl::SpeculativeDecodingImpl(
    const std::string& main_models_path,
    const std::string& draft_models_path,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config) {
    m_main_pipeline = std::make_shared<ContinuousBatchingImpl>(main_models_path, tokenizer, scheduler_config, device, plugin_config, true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingImpl>(draft_models_path, tokenizer, scheduler_config, device, plugin_config, false);
}

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const ov::Tensor& input_ids,
                                                                 ov::genai::GenerationConfig sampling_params) {
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params);
};

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const std::string& prompt,
                                                                 ov::genai::GenerationConfig sampling_params) {
    return m_main_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::SpeculativeDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::step() {
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::SpeculativeDecodingImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                              const std::vector<GenerationConfig>& sampling_params,
                                                              const StreamerVariant& streamer) {
    return m_main_pipeline->generate(input_ids, sampling_params, streamer);
}

std::vector<GenerationResult>
ContinuousBatchingPipeline::SpeculativeDecodingImpl::generate(const std::vector<std::string>& prompts,
                                                              std::vector<ov::genai::GenerationConfig> sampling_params,
                                                              const StreamerVariant& streamer) {
    return m_main_pipeline->generate(prompts, sampling_params, streamer);
}

}
