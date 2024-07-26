// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/openvino.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/generation_handle.hpp"

namespace ov::genai {
struct PipelineMetrics { 
    // All requests as viewed by the pipeline
    size_t requests = 0;
    // Requests scheduled for processing
    size_t scheduled_requests = 0;
    // Percentage of KV cache usage
    float cache_usage = 0.0;
};

class OPENVINO_GENAI_EXPORTS BasicPipeline {
protected:
    ov::genai::Tokenizer m_tokenizer;

    ov::Tensor encode(const std::string& prompt);
    std::vector<ov::Tensor> encode(const std::vector<std::string>& prompts);

    std::string decode(const std::vector<int64_t>& line);

    std::vector<GenerationResult>
    process_generated_sequences(const std::vector<GenerationHandle>& generations,
                                std::vector<ov::genai::GenerationConfig> sampling_params);

    virtual GenerationHandle add_request(uint64_t request_id, ov::Tensor tokenized_prompt, ov::genai::GenerationConfig sampling_params) = 0;
    virtual std::vector<GenerationHandle> generate_sequences(
        const std::vector<ov::Tensor> prompts, std::vector<ov::genai::GenerationConfig> sampling_params) = 0;


public:
    // more high level interface, which can process multiple prompts in continuous batching manner
    std::vector<GenerationResult> generate(
        const std::vector<std::string>& prompts, std::vector<ov::genai::GenerationConfig> sampling_params);

    ov::genai::Tokenizer get_tokenizer();

    GenerationHandle add_request(uint64_t request_id, std::string prompt, ov::genai::GenerationConfig sampling_params);

    virtual PipelineMetrics get_metrics() const = 0;

    virtual void step() = 0;

    virtual bool has_non_finished_requests() = 0;
};
} // namespace ov::genai
