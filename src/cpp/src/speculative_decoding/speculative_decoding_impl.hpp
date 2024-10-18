// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "continuous_batching/continuous_batching_impl.hpp"

#include "speculative_decoding/continuous_batching_for_speculative_decoding_impl.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"

namespace ov::genai {

struct ModelDesc {
    std::string model_path;
    std::string device;
    ov::genai::SchedulerConfig scheduler_config;
    ov::AnyMap plugin_config;

    ModelDesc(const std::string& model_path,
              const std::string& device = "",
              const ov::AnyMap& plugin_config = {},
              const ov::genai::SchedulerConfig& scheduler_config = {}) :
        model_path(model_path),
        device(device),
        plugin_config(plugin_config),
        scheduler_config(scheduler_config) {}
};

extern const std::string DRAFT_MODEL_ARG_NAME;

inline ov::genai::ModelDesc
extract_draft_model_from_config(ov::AnyMap& config) {
    ov::genai::ModelDesc draft_model("");
    auto it = config.find(DRAFT_MODEL_ARG_NAME);
    if (it != config.end()) {
        draft_model = it->second.as<ov::genai::ModelDesc>();
        config.erase(it);
    }
    return draft_model;
}

class ContinuousBatchingPipeline::SpeculativeDecodingImpl : public ContinuousBatchingPipeline::ImplInterface {
protected:
    std::shared_ptr<ContinuousBatchingForSpeculativeDecodingImpl> m_main_pipeline, m_draft_pipeline;
    SpeculativeDecodingMetrics m_sd_metrics;
    
public:
    SpeculativeDecodingImpl(const std::string& main_models_path,
                            const SchedulerConfig& scheduler_config,
                            const std::string& device,
                            const ov::AnyMap& plugin_config,
                            const ov::genai::ModelDesc draft_model_desc,
                            const ov::AnyMap& tokenizer_config = {});

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