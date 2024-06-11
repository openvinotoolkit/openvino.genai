// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <openvino/openvino.hpp>

#include "scheduler_config.hpp"
#include "tokenizer.hpp"
#include "generation_config.hpp"
#include "generation_handle.hpp"

using plugin_config_t = std::map<std::string, ov::Any>;

class ContinuousBatchingPipeline {
    class Impl;
    std::shared_ptr<Impl> m_impl;

public:
    ContinuousBatchingPipeline(const std::string& models_path,
                               const SchedulerConfig& scheduler_config,
                               const std::string& device = "CPU",
                               const std::string& plugin_config = {});

    std::shared_ptr<Tokenizer> get_tokenizer();

    GenerationConfig get_config() const;

    const plugin_config_t& get_plugin_config();

    GenerationHandle add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params);

    void step();

    bool has_non_finished_requests();

    // more high level interface, which can process multiple prompts in continuous batching manner
    std::vector<GenerationResult> generate(const std::vector<std::string>& prompts, std::vector<GenerationConfig> sampling_params);

private:
    bool parse_plugin_config(std::string config_string);

    bool parse_plugin_config(const nlohmann::json& node);
};
