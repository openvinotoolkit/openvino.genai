// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "continuous_batching_impl.hpp"

namespace ov::genai {
class ContinuousBatchingPipeline::SpeculativeDecodingImpl : public ContinuousBatchingPipeline::ImplInterface {
protected:
    std::shared_ptr<ContinuousBatchingImpl> m_main_pipeline, m_draft_pipeline;
    size_t m_num_candidates = 5, m_max_num_candidates = 10;
    std::map<int64_t, size_t> m_to_generate_length;

    void update_strategy(size_t num_matches);
    
public:
    SpeculativeDecodingImpl(const std::string& main_models_path,
                            const std::string& draft_models_path,
                            const Tokenizer& tokenizer,
                            const SchedulerConfig& scheduler_config,
                            const std::string& device,
                            const ov::AnyMap& plugin_config);

    SpeculativeDecodingImpl(const std::string& main_models_path,
                            const std::string& draft_models_path,
                            const SchedulerConfig& scheduler_config,
                            const std::string& device,
                            const ov::AnyMap& llm_plugin_config,
                            const ov::AnyMap& tokenizer_plugin_config)
    : SpeculativeDecodingImpl{main_models_path, draft_models_path, Tokenizer(main_models_path, tokenizer_plugin_config), scheduler_config, device, llm_plugin_config} {};


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
    // std::vector<GenerationResult>
    // generate(const std::vector<std::string>& prompts,
    //          std::vector<ov::genai::GenerationConfig> sampling_params,
    //          const StreamerVariant& streamer) override;
};
}