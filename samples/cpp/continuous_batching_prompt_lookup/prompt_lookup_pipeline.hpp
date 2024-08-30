// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/cb_basic_pipeline.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

class PromptLookupPipeline : public ov::genai::BasicPipeline {
    ov::genai::ContinuousBatchingPipeline model_pipeline;
    size_t candidates_number = 0, max_candidates_number = 0, max_ngram_size = 0;
    bool is_speculative_mode = false;
    std::map<uint64_t, std::vector<int64_t>> m_encoded_prompt;

    std::vector<ov::genai::GenerationHandle> generate_sequences(
        const std::vector<ov::Tensor> prompts, std::vector<ov::genai::GenerationConfig> sampling_params) override;


public:
    size_t infer_cnt = 0, max_matches = 0, avg_matches = 0;

    PromptLookupPipeline(const std::string& models_path,
                         size_t candidates_number,
                         size_t max_ngram_size,
                         const ov::genai::SchedulerConfig& scheduler_config,
                         const std::string& device = "CPU",
                         const ov::AnyMap& plugin_config = {});

    PromptLookupPipeline(const std::string& models_path,
                         size_t candidates_number,
                         size_t max_ngram_size,
                         const ov::genai::Tokenizer& tokenizer,
                         const ov::genai::SchedulerConfig& scheduler_config,
                         const std::string& device = "CPU",
                         const ov::AnyMap& plugin_config = {});

    ov::genai::PipelineMetrics get_metrics() const override;

    void step() override;

    bool has_non_finished_requests() override;

    void set_k(size_t new_default_k);

    void update_strategy(size_t num_matches);
};