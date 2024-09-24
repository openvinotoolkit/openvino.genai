// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "continuous_batching_impl_interface.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

namespace ov::genai {
class ContinuousBatchingPipeline::ContinuousBatchingImpl : public ContinuousBatchingPipeline::ImplInterface {
protected:
    void _free_non_running_requests();
    void _notify_requests_dropped_by_handle();

public:
    ContinuousBatchingImpl(const std::string& models_path,
                           const Tokenizer& tokenizer,
                           const SchedulerConfig& scheduler_config,
                           const std::string& device,
                           const ov::AnyMap& plugin_config);

    ContinuousBatchingImpl(const std::string& models_path,
                           const SchedulerConfig& scheduler_config,
                           const std::string& device,
                           const ov::AnyMap& llm_plugin_config,
                           const ov::AnyMap& tokenizer_plugin_config)
    : ContinuousBatchingImpl{models_path, Tokenizer(models_path, tokenizer_plugin_config), scheduler_config, device, llm_plugin_config} {};


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
    std::vector<GenerationResult>
    generate(const std::vector<std::string>& prompts,
             std::vector<ov::genai::GenerationConfig> sampling_params,
             const StreamerVariant& streamer) override;
};
}