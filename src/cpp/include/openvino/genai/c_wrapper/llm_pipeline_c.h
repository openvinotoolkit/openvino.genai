// Copyright (C) 2025 Intel Corporation
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

    /**
     * @brief Create DecodedResults
     */
    OPENVINO_GENAI_EXPORTS DecodedResultsHandle* CreateDecodedResults();

    /**
     * @brief Release the memory allocated by DecodedResultsHandle.
     * @param model A pointer to the DecodedResultsHandle to free memory.
     */
    OPENVINO_GENAI_EXPORTS void DestroyDecodedResults(DecodedResultsHandle * results);

    /**
     * @brief Get performance metrics from DecodedResultsHandle.
     * @param results A pointer to the DecodedResultsHandle.
     * @param metrics A pointer to the PerfMetricsHandle.
     */
    OPENVINO_GENAI_EXPORTS void DecodedeResultsGetPerfMetrics(DecodedResultsHandle * results,
                                                              PerfMetricsHandle ** metrics);

    /**
     * @brief Get string result from DecodedResultsHandle.
     * @param results A pointer to the DecodedResultsHandle.
     * @param output A pointer to the output string buffer.
     * @param max_size The maximum size of the output buffer.
     */
    OPENVINO_GENAI_EXPORTS void DecodeResultsGetString(DecodedResultsHandle * results, char* output, int max_size);

    /**
     * @struct LLMPipelineHandle
     * @brief type define LLMPipelineHandle from OpaqueLLMPipeline
     */
    typedef struct OpaqueLLMPipeline LLMPipelineHandle;

    /**
     * @brief Construct LLMPipelineHandle.
     */
    OPENVINO_GENAI_EXPORTS LLMPipelineHandle* CreateLLMPipeline(const char* models_path, const char* device);

    /**
     * @brief Release the memory allocated by LLMPipelineHandle.
     * @param model A pointer to the LLMPipelineHandle.
     */
    OPENVINO_GENAI_EXPORTS void DestroyLLMPipeline(LLMPipelineHandle * pipe);

    /**
     * @brief Generate text by LLMPipelineHandle.
     * @param pipe A pointer to the LLMPipelineHandle.
     * @param inputs A pointer to the input string.
     * @param output A pointer to the output string buffer.
     * @param max_size The maximum size of the output buffer.
     * @param config A pointer to the GenerationConfigHandle, the pointer can be NULL.
     */
    OPENVINO_GENAI_EXPORTS void LLMPipelineGenerate(LLMPipelineHandle * handle,
                                                    const char* inputs,
                                                    char* output,
                                                    int max_size,
                                                    GenerationConfigHandle* config);

    /*@brief Generate text by LLMPipelineHandle with Streamer.
     * @param pipe A pointer to the LLMPipelineHandle.
     * @param inputs A pointer to the input string.
     * @param output A pointer to the output string buffer.
     * @param max_size The maximum size of the output buffer.
     * @param config A pointer to the GenerationConfigHandle, the pointer can be NULL.
     * @param buffer A pointer to the stream buffer.
     * @param buffer_size The size of the stream buffer.
     * @param buffer_pos A pointer to the stream buffer position.
     */
    OPENVINO_GENAI_EXPORTS void LLMPipelineGenerateStream(LLMPipelineHandle * pipe,
                                                          const char* inputs,
                                                          char* output,
                                                          int max_size,
                                                          GenerationConfigHandle* config,
                                                          char* buffer,
                                                          const int buffer_size,
                                                          int* buffer_pos);

    /**
     * @brief Generate text by LLMPipelineHandle and return DecodedResultsHandle.
     * @param pipe A pointer to the LLMPipelineHandle.
     * @param inputs A pointer to the input string.
     * @pram config A pointer to the GenerationConfigHandle, the pointer can be NULL.
     * @return DecodedResultsHandle A pointer to the DecodedResultsHandle.
     */
    OPENVINO_GENAI_EXPORTS DecodedResultsHandle* LLMPipelineGenerateDecodeResults(LLMPipelineHandle * handle,
                                                                                  const char* inputs,
                                                                                  GenerationConfigHandle* config);
    /**
     * @brief Start chat with keeping history in kv cache.
     * @param pipe A pointer to the LLMPipelineHandle.
     */
    OPENVINO_GENAI_EXPORTS void LLMPipelineStartChat(LLMPipelineHandle * pipe);

    /**
     * @brief Finish chat and clear kv cache.
     * @param pipe A pointer to the LLMPipelineHandle.
     */
    OPENVINO_GENAI_EXPORTS void LLMPipelineFinishChat(LLMPipelineHandle * pipe);

    /**
     * @brief Get the GenerationConfig from LLMPipelineHandle.
     * @param pipe A pointer to the LLMPipelineHandle.
     * @return GenerationConfigHandle A pointer to the GenerationConfigHandle.
     */
    OPENVINO_GENAI_EXPORTS GenerationConfigHandle* LLMPipelineGetGeneratonConfig(LLMPipelineHandle * pipe);

    /**
     * @brief Set the GenerationConfig to LLMPipelineHandle.
     * @param pipe A pointer to the LLMPipelineHandle.
     * @param config A pointer to the GenerationConfigHandle.
     */
    OPENVINO_GENAI_EXPORTS void LLMPipelineSetGeneratonConfig(LLMPipelineHandle * pipe,
                                                              GenerationConfigHandle * config);
    // TODO: Add C wrapper for class EncodedResults and LLMPipeline::generation with Streamer.
#ifdef __cplusplus
}
#endif
