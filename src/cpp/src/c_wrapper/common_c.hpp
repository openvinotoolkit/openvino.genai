// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/visibility.hpp"

#ifdef __cplusplus
OPENVINO_EXTERN_C {
#endif

    /**
     * @struct OpaqueGenerationConfig
     * @brief This is an interface of ov::genai::GenerationConfig
     */
    struct OpaqueGenerationConfig {
        std::shared_ptr<ov::genai::GenerationConfig> object;
    };

    /**
     * @struct OpaqueLLMPipeline
     * @brief This is an interface of ov::genai::LLMPipeline
     */
    struct OpaqueLLMPipeline {
        std::shared_ptr<ov::genai::LLMPipeline> object;
    };
    /**
     * @struct OpaquePerfMetrics
     * @brief This is an interface of ov::genai::PerfMetrics
     */
    struct OpaquePerfMetrics {
        std::shared_ptr<ov::genai::PerfMetrics> object;
    };
    /**
     * @struct OpaqueDecodedResults
     * @brief This is an interface of ov::genai::DecodedResults
     */
    struct OpaqueDecodedResults {
        std::shared_ptr<ov::genai::DecodedResults> object;
    };

#ifdef __cplusplus
}
#endif
