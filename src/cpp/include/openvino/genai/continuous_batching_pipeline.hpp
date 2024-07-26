// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include <memory>
// #include <openvino/openvino.hpp>

#include "openvino/genai/scheduler_config.hpp"
// #include "openvino/genai/tokenizer.hpp"
// #include "openvino/genai/generation_config.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/visibility.hpp"

#include "openvino/genai/cb_basic_pipeline.hpp"

namespace ov::genai {
class OPENVINO_GENAI_EXPORTS ContinuousBatchingPipeline : public ov::genai::BasicPipeline {
protected:
    class Impl;
    std::shared_ptr<Impl> m_impl;

    // GenerationHandle add_request(uint64_t request_id, ov::Tensor tokenized_prompt, ov::genai::GenerationConfig sampling_params) override;

    std::vector<GenerationHandle> generate_sequences(
        const std::vector<ov::Tensor> prompts, std::vector<ov::genai::GenerationConfig> sampling_params) override;

public:
    ContinuousBatchingPipeline(const std::string& models_path,
                               const SchedulerConfig& scheduler_config,
                               const std::string& device = "CPU",
                               const ov::AnyMap& plugin_config = {});

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

    // todo: iefode
    ContinuousBatchingPipeline() = default;

    PipelineMetrics get_metrics() const override;

    void step() override;

    bool has_non_finished_requests() override;

    GenerationHandle add_request(uint64_t request_id, std::string prompt, ov::genai::GenerationConfig sampling_params);
    GenerationHandle add_request(uint64_t request_id, ov::Tensor prompt, ov::genai::GenerationConfig sampling_params);

};
}
