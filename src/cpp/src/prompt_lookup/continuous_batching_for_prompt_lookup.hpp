// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"

#include "continuous_batching_impl.hpp"

namespace ov::genai {
class ContinuousBatchingPipeline::ContinuousBatchingForPromptLookupImpl : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
public:
    ContinuousBatchingForPromptLookupImpl() = default;

    ContinuousBatchingForPromptLookupImpl(
        const std::filesystem::path& models_path,
        const Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties) :
        ContinuousBatchingImpl{ models_path, tokenizer, scheduler_config, device, properties} {};
    
    void step() override;
};
}