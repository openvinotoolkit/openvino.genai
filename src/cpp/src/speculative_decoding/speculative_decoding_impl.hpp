// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "continuous_batching/continuous_batching_impl.hpp"
#include "speculative_decoding/continuous_batching_for_speculative_decoding_impl.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"

namespace ov::genai {

struct ModelDesc {
    std::string device;
    ov::genai::SchedulerConfig scheduler_config;
    ov::AnyMap properties;
    ov::genai::GenerationConfig generation_config;
    std::shared_ptr<ov::Model> model = nullptr;
    ov::genai::Tokenizer tokenizer;

    ModelDesc(const std::shared_ptr<ov::Model>& model,
              const ov::genai::Tokenizer& tokenizer,
              const std::string& device = {},
              const ov::AnyMap& properties = {},
              const ov::genai::SchedulerConfig& scheduler_config = {},
              const ov::genai::GenerationConfig& generation_config = {}) :
        model(model),
        tokenizer(tokenizer),
        device(device),
        properties(properties),
        scheduler_config(scheduler_config),
        generation_config(generation_config) {}
    
    ModelDesc() = default;
};

class ContinuousBatchingPipeline::SpeculativeDecodingImpl : public ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    std::shared_ptr<ContinuousBatchingForSpeculativeDecodingImpl> m_main_pipeline, m_draft_pipeline;
    // Metrics
    SpeculativeDecodingMetrics m_sd_metrics;
    PerfMetrics m_perf_metrics;

    // Mutex protecting access to m_draft_generations, so add_request and step methods can be called from different threads
    std::mutex m_draft_generations_mutex;
    std::map<uint64_t, GenerationHandle> m_draft_generations;

    void drop_requests();
    bool is_requests_empty();
    std::vector<SequenceGroup::Ptr> get_awaiting_requests();
    
public:
    SpeculativeDecodingImpl(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc);

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

    SpeculativeDecodingMetrics get_speculative_decoding_metrics();
};

}
