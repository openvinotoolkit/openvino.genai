// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/openvino.hpp>

#include "scheduler_config.hpp"
#include "tokenizer.hpp"
#include "generation_config.hpp"
#include "generation_handle.hpp"

class ContinuousBatchingPipeline {
    class Impl;
    std::shared_ptr<Impl> m_impl;

public:
    ContinuousBatchingPipeline(const std::string& models_path,
                               const SchedulerConfig& scheduler_config,
                               const std::string& device = "CPU",
                               const ov::AnyMap& plugin_config = {});

    std::shared_ptr<Tokenizer> get_tokenizer();

    GenerationConfig get_config() const;

    GenerationHandle add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params);

    void step();

    bool has_non_finished_requests();

    // more high level interface, which can process multiple prompts in continuous batching manner
    std::vector<GenerationResult> generate(const std::vector<std::string>& prompts, std::vector<GenerationConfig> sampling_params);
};
