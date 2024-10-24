// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "continuous_batching_impl.hpp"
#include "continuous_batching_for_speculative_decoding_impl.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"

namespace ov::genai {

struct ModelDesc {
    std::filesystem::path models_path;
    std::string device;
    ov::genai::SchedulerConfig scheduler_config;
    ov::AnyMap properties;

    ModelDesc(const std::filesystem::path& models_path,
              const std::string& device = {},
              const ov::AnyMap& properties = {},
              const ov::genai::SchedulerConfig& scheduler_config = {}) :
        models_path(models_path),
        device(device),
        properties(properties),
        scheduler_config(scheduler_config) {}
};

class ContinuousBatchingPipeline::SpeculativeDecodingImpl : public ContinuousBatchingPipeline::ImplInterface {
protected:
    std::shared_ptr<ContinuousBatchingForSpeculativeDecodingImpl> m_main_pipeline, m_draft_pipeline;
    SpeculativeDecodingMetrics m_sd_metrics;
    // Mutex protecting access to m_draft_generations, so add_request and step methods can be called from different threads
    std::mutex m_draft_generations_mutex;
    std::map<uint64_t, GenerationHandle> m_draft_generations;
    
public:
    SpeculativeDecodingImpl(const std::filesystem::path& main_models_path,
                            const SchedulerConfig& scheduler_config,
                            const std::string& device,
                            const ov::AnyMap& properties,
                            const ov::genai::ModelDesc draft_model_desc,
                            const ov::AnyMap& tokenizer_properties = {});

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