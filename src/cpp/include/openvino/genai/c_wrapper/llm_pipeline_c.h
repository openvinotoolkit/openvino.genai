// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for  ov::genai::LLMPipeline class.
 *
 * @file llm_pipeline_c.h
 */

#pragma once
#include "generation_config_c.h"
#include "perf_metrics_c.h"
#ifdef __cplusplus
OPENVINO_EXTERN_C {
#endif
    /**
     * @struct DecodedResultsHandle
     * @brief type define DecodedResultsHandle from OpaqueDecodedResults
     */
    typedef struct OpaqueDecodedResults DecodedResultsHandle;
    OPENVINO_GENAI_EXPORTS DecodedResultsHandle* CreateDecodedResults();
    OPENVINO_GENAI_EXPORTS void DestroyDecodedResults(DecodedResultsHandle * results);
    OPENVINO_GENAI_EXPORTS void DecodedeResultsGetPerfMetrics(DecodedResultsHandle * results,
                                                              PerfMetricsHandle * metrics);
    OPENVINO_GENAI_EXPORTS void DecodeResultsGetString(DecodedResultsHandle * results, char* output, int max_size);

     /**
     * @struct LLMPipelineHandle
     * @brief type define LLMPipelineHandle from OpaqueLLMPipeline
     */
    typedef struct OpaqueLLMPipeline LLMPipelineHandle;
    OPENVINO_GENAI_EXPORTS LLMPipelineHandle* CreateLLMPipeline(const char* models_path, const char* device);
    OPENVINO_GENAI_EXPORTS void DestroyLLMPipeline(LLMPipelineHandle * pipe);
    OPENVINO_GENAI_EXPORTS void LLMPipelineGenerate(LLMPipelineHandle * handle,
                                                    const char* inputs,
                                                    char* output,
                                                    int max_size,
                                                    GenerationConfigHandle* config);

    OPENVINO_GENAI_EXPORTS DecodedResultsHandle* LLMPipelineGenerateDecodeResults(LLMPipelineHandle * handle,
                                                                 const char* inputs,
                                                                 GenerationConfigHandle* config);
    OPENVINO_GENAI_EXPORTS void LLMPipelineStartChat(LLMPipelineHandle * pipe);
    OPENVINO_GENAI_EXPORTS void LLMPipelineFinishChat(LLMPipelineHandle * pipe);

    OPENVINO_GENAI_EXPORTS GenerationConfigHandle* LLMPipelineGetGeneratonConfig(LLMPipelineHandle * pipe);
    OPENVINO_GENAI_EXPORTS void LLMPipelineSetGeneratonConfig(LLMPipelineHandle * pipe,
                                                              GenerationConfigHandle * config);
    //TODO: Add C wrapper for class EncodedResults and LLMPipeline::generation with Streamer.
#ifdef __cplusplus
}
#endif
