// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/openvino.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

class SpeculativeDecodingPipeline {
protected:
    ov::genai::Tokenizer m_tokenizer;
    ov::genai::ContinuousBatchingPipeline m_pipeline, m_speculative_pipeline;

    bool m_is_speculative_mode = false;
    size_t m_max_candidates_num = 0, m_candidates_num = 0;

    void update_strategy(size_t num_matches);

public:
    SpeculativeDecodingPipeline(const std::string& models_path,
                                const std::string& speculative_model_path,
                                size_t start_candidates_number,
                                const ov::genai::SchedulerConfig& scheduler_config,
                                const std::string& device = "CPU",
                                const ov::AnyMap& plugin_config = {});

    void step();

    std::vector<ov::genai::GenerationResult> generate(const std::vector<std::string>& prompts, const std::vector<ov::genai::GenerationConfig>& sampling_params);

    size_t m_max_matches = 0;
    std::vector<size_t> m_matches_info;
    int64_t m_speculative_model_duration = 0;
};