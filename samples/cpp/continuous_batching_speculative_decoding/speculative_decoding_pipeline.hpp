// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/cb_basic_pipeline.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

class SpeculativeDecodingPipeline : public ov::genai::BasicPipeline {
    ov::genai::ContinuousBatchingPipeline model_pipeline, assisting_pipeline;
    size_t k = 0, default_k = 0;
    bool is_speculative_mode = false;

    std::vector<ov::genai::GenerationHandle> generate_sequences(
        const std::vector<ov::Tensor> prompts, std::vector<ov::genai::GenerationConfig> sampling_params) override;


public:
    SpeculativeDecodingPipeline(const std::string& models_path,
                                const std::string& assisting_model_path,
                                const ov::genai::SchedulerConfig& scheduler_config,
                                const std::string& device = "CPU",
                                const ov::AnyMap& plugin_config = {});

    SpeculativeDecodingPipeline(const std::string& models_path,
                                const std::string& assisting_model_path,
                                const ov::genai::Tokenizer& tokenizer,
                                const ov::genai::SchedulerConfig& scheduler_config,
                                const std::string& device = "CPU",
                                const ov::AnyMap& plugin_config = {});

    ov::genai::PipelineMetrics get_metrics() const override;

    void step() override;

    bool has_non_finished_requests() override;
    
    void set_k(size_t new_default_k);
};