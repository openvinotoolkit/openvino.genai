// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/visibility.hpp"

/**
 * @struct ov_genai_generation_config_opaque
 * @brief This is an interface of ov::genai::GenerationConfig
 */
struct ov_genai_generation_config_opaque {
    std::shared_ptr<ov::genai::GenerationConfig> object;
};

/**
 * @struct ov_genai_llm_pipeline_opaque
 * @brief This is an interface of ov::genai::LLMPipeline
 */
struct ov_genai_llm_pipeline_opaque {
    std::shared_ptr<ov::genai::LLMPipeline> object;
};
/**
 * @struct ov_genai_perf_metrics_opaque
 * @brief This is an interface of ov::genai::PerfMetrics
 */
struct ov_genai_perf_metrics_opaque {
    std::shared_ptr<ov::genai::PerfMetrics> object;
};
/**
 * @struct ov_genai_decoded_results_opaque
 * @brief This is an interface of ov::genai::DecodedResults
 */
struct ov_genai_decoded_results_opaque {
    std::shared_ptr<ov::genai::DecodedResults> object;
};
