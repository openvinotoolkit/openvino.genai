// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/openvino.hpp>

#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {
struct PipelineMetrics { 
    // All requests as viewed by the pipeline
    size_t requests = 0;
    // Requests scheduled for processing
    size_t scheduled_requests = 0;
    // Percentage of KV cache usage
    float cache_usage = 0.0;
};

class OPENVINO_GENAI_EXPORTS ContinuousBatchingPipeline {
    class Impl;
    std::shared_ptr<Impl> m_impl;

public:
    ContinuousBatchingPipeline(const std::string& models_path,
                               const SchedulerConfig& scheduler_config,
                               const std::string& device = "CPU",
                               const ov::AnyMap& llm_plugin_config = {},
                               const ov::AnyMap& tokenizer_plugin_config = {});

    /**
    * @brief Constructs a ContinuousBatchingPipeline when ov::genai::Tokenizer is initialized manually using file from the different dirs.
    *
    * @param model_path Path to the dir with model, tokenizer .xml/.bin files, and generation_configs.json
    * @param scheduler_config
    * @param tokenizer manually initialized ov::genai::Tokenizer
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    ContinuousBatchingPipeline(
        const std::string& model_path,
        const ov::genai::Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device="CPU",
        const ov::AnyMap& plugin_config={}
    );

    ov::genai::Tokenizer get_tokenizer();

    ov::genai::GenerationConfig get_config() const;

    PipelineMetrics get_metrics() const;

    GenerationHandle add_request(uint64_t request_id, std::string prompt, ov::genai::GenerationConfig sampling_params);

    void step();

    bool has_non_finished_requests();

    // more high level interface, which can process multiple prompts in continuous batching manner
    std::vector<GenerationResult> generate(const std::vector<std::string>& prompts, std::vector<ov::genai::GenerationConfig> sampling_params);
};
}
