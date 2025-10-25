// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "continuous_batching_for_prompt_lookup.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "utils.hpp"

namespace ov::genai {

class ContinuousBatchingPipeline::PromptLookupImpl : public ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    std::shared_ptr<ContinuousBatchingForPromptLookupImpl> m_pipeline;
    SpeculativeDecodingMetrics m_sd_metrics;
    PerfMetrics m_perf_metrics;

    void drop_requests();

public:
    PromptLookupImpl(const std::shared_ptr<ov::Model>& model,
                     const Tokenizer& tokenizer,
                     const SchedulerConfig& scheduler_config,
                     const std::string& device,
                     const ov::AnyMap& properties,
                     const ov::genai::GenerationConfig& generation_config) {
        m_tokenizer = tokenizer;
        m_perf_metrics.raw_metrics.m_inference_durations = {{ MicroSeconds(0.0f) }};
        m_pipeline = std::make_shared<ContinuousBatchingForPromptLookupImpl>(model, tokenizer, scheduler_config, device, properties, generation_config);
    };

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt) override;
    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

    bool has_non_finished_requests() override;

    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             std::optional<std::vector<ov::Tensor>> token_type_ids = std::nullopt) override;

    SpeculativeDecodingMetrics get_metrics();
};

}
