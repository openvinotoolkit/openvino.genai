// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "continuous_batching_impl.hpp"
#include "continuous_batching_for_prompt_lookup.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"

namespace ov::genai {

class ContinuousBatchingPipeline::PromptLookupImpl : public ContinuousBatchingPipeline::ImplInterface {
protected:
    std::shared_ptr<ContinuousBatchingForPromptLookupImpl> m_pipeline;
    SpeculativeDecodingMetrics m_sd_metrics;
    
public:
    PromptLookupImpl(
        const std::filesystem::path& models_path,
        const Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties) {
        m_pipeline = std::make_shared<ContinuousBatchingForPromptLookupImpl>(models_path, tokenizer, scheduler_config, device, properties);
        m_tokenizer = tokenizer;
    };

    PromptLookupImpl(
        const std::filesystem::path& models_path,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties,
        const ov::AnyMap& tokenizer_properties = {}) {
        m_tokenizer = Tokenizer(models_path, tokenizer_properties);
        m_pipeline = std::make_shared<ContinuousBatchingForPromptLookupImpl>(models_path, m_tokenizer, scheduler_config, device, properties);
    };

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 ov::genai::GenerationConfig sampling_params) override;
    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 ov::genai::GenerationConfig sampling_params) override;

    bool has_non_finished_requests() override;

    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer) override;

    SpeculativeDecodingMetrics get_metrics();
};

}