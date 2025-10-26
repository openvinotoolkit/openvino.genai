// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for ov::genai::VLMPipeline class.
 *
 * @file vlm_pipeline.h
 */

#pragma once
#include "generation_config.h"
#include "perf_metrics.h"
#include "openvino/c/ov_common.h"
#include "openvino/c/ov_tensor.h"
#include "openvino/genai/c/visibility.h"

/**
 * @struct ov_genai_vlm_decoded_results
 * @brief type define ov_genai_vlm_decoded_results from ov_genai_vlm_decoded_results_opaque
 */
typedef struct ov_genai_vlm_decoded_results_opaque ov_genai_vlm_decoded_results;

/**
 * @brief Create VLMDecodedResults
 * @param results A pointer to the newly created ov_genai_vlm_decoded_results.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_vlm_decoded_results_create(ov_genai_vlm_decoded_results** results);

/**
 * @brief Release the memory allocated by ov_genai_vlm_decoded_results.
 * @param results A pointer to the ov_genai_vlm_decoded_results to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_vlm_decoded_results_free(ov_genai_vlm_decoded_results* results);

/**
 * @brief Get performance metrics from ov_genai_vlm_decoded_results.
 * @param results A pointer to the ov_genai_vlm_decoded_results instance.
 * @param metrics A pointer to the newly created ov_genai_perf_metrics.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_vlm_decoded_results_get_perf_metrics(const ov_genai_vlm_decoded_results* results,
                                                                                   ov_genai_perf_metrics** metrics);

/**
 * @brief Release the memory allocated by ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_vlm_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics);

/**
 * @brief Get string result from ov_genai_vlm_decoded_results.
 * @param results A pointer to the ov_genai_vlm_decoded_results instance.
 * @param output A pointer to the pre-allocated output string buffer. It can be set to NULL, in which case the
 * *output_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire output.
 * @param output_size A Pointer to the size of the output string from the results, including the null terminator. If
 * output is not NULL, *output_size should be greater than or equal to the result string size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_vlm_decoded_results_get_string(const ov_genai_vlm_decoded_results* results,
                                                                             char* output,
                                                                             size_t* output_size);

/**
 * @struct ov_genai_vlm_pipeline
 * @brief type define ov_genai_vlm_pipeline from ov_genai_vlm_pipeline_opaque
 */
typedef struct ov_genai_vlm_pipeline_opaque ov_genai_vlm_pipeline;

/**
 * @brief Construct ov_genai_vlm_pipeline.
 *
 * Initializes a ov_genai_vlm_pipeline instance from the specified model directory and device. Optional property
 * parameters can be passed as key-value pairs.
 *
 * @param models_path Path to the directory containing the model files.
 * @param device Name of a device to load a model to.
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param pipe A pointer to the newly created ov_genai_vlm_pipeline.
 * @param ... property parameter: Optional pack of pairs: <char* property_key, char* property_value> relevant only
 * @return ov_status_e A status code, return OK(0) if successful.
 *
 * @example
 * Example with no properties:
 * ov_genai_vlm_pipeline_create(model_path, "CPU", 0, &pipe);
 *
 * Example with properties:
 * ov_genai_vlm_pipeline_create(model_path, "GPU", 2, &pipe,
 *                             "CACHE_DIR", "cache_dir");
 * Note: If the property value is not a string, please pass it as a string
 * Example:
 * ov_genai_vlm_pipeline_create(model_path, "NPU", 6, &pipeline, "MAX_PROMPT_LEN", "128", "MIN_RESPONSE_LEN",
 *                             "64", "CACHE_DIR", "cache_dir")
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_vlm_pipeline_create(const char* models_path,
                                                                  const char* device,
                                                                  const size_t property_args_size,
                                                                  ov_genai_vlm_pipeline** pipe,
                                                                  ...);

/**
 * @brief Release the memory allocated by ov_genai_vlm_pipeline.
 * @param pipe A pointer to the ov_genai_vlm_pipeline to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_vlm_pipeline_free(ov_genai_vlm_pipeline* pipe);

#ifndef OV_GENAI_STREAMING_STATUS_DEFINED
#define OV_GENAI_STREAMING_STATUS_DEFINED
typedef enum {
    OV_GENAI_STREAMING_STATUS_RUNNING = 0,  // Continue to run inference
    OV_GENAI_STREAMING_STATUS_STOP =
        1,  // Stop generation, keep history as is, KV cache includes last request and generated tokens
    OV_GENAI_STREAMING_STATUS_CANCEL = 2  // Stop generate, drop last prompt and all generated tokens from history, KV
                                           // cache includes history but last step
} ov_genai_streaming_status_e;

#endif // OV_GENAI_STREAMING_STATUS_DEFINED

#ifndef OV_GENAI_STREAMER_CALLBACK_DEFINED
#define OV_GENAI_STREAMER_CALLBACK_DEFINED
/**
 * @brief Structure for streamer callback functions with arguments.
 *
 * The callback function takes two parameters:
 * - `const char* str`: A constant string extracted from the decoded result for processing
 * - `void* args`: A pointer to additional arguments, allowing flexible data passing.
 */
typedef struct {
    ov_genai_streaming_status_e(
        OPENVINO_C_API_CALLBACK* callback_func)(const char* str, void* args);  //!< Pointer to the callback function
    void* args;  //!< Pointer to the arguments passed to the callback function
} streamer_callback;
#endif // OV_GENAI_STREAMER_CALLBACK_DEFINED

/**
 * @brief Generate results by ov_genai_vlm_pipeline with text and image inputs
 * @param pipe A pointer to the ov_genai_vlm_pipeline instance.
 * @param text_inputs A pointer to the input text string.
 * @param rgbs A pointer to the array of ov_tensor_t containing image data.
 * @param num_images Number of images in the rgbs array.
 * @param config A pointer to the ov_genai_generation_config, the pointer can be NULL.
 * @param streamer A pointer to the stream callback. Set to NULL if no callback is needed. Either this or results must
 * be non-NULL.
 * @param results A pointer to the ov_genai_vlm_decoded_results, which retrieves the results of the generation. Either this
 * or streamer must be non-NULL.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_vlm_pipeline_generate(ov_genai_vlm_pipeline* pipe,
                                                                    const char* text_inputs,
                                                                    const ov_tensor_t** rgbs,
                                                                    size_t num_images,
                                                                    const ov_genai_generation_config* config,
                                                                    const streamer_callback* streamer,
                                                                    ov_genai_vlm_decoded_results** results);

/**
 * @brief Start chat with keeping history in kv cache.
 * @param pipe A pointer to the ov_genai_vlm_pipeline instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_vlm_pipeline_start_chat(ov_genai_vlm_pipeline* pipe);

/**
 * @brief Finish chat and clear kv cache.
 * @param pipe A pointer to the ov_genai_vlm_pipeline instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_vlm_pipeline_finish_chat(ov_genai_vlm_pipeline* pipe);

/**
 * @brief Get the GenerationConfig from ov_genai_vlm_pipeline.
 * @param pipe A pointer to the ov_genai_vlm_pipeline instance.
 * @param config A pointer to the newly created ov_genai_generation_config.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_vlm_pipeline_get_generation_config(const ov_genai_vlm_pipeline* pipe,
                                                                                 ov_genai_generation_config** config);

/**
 * @brief Set the GenerationConfig to ov_genai_vlm_pipeline.
 * @param pipe A pointer to the ov_genai_vlm_pipeline instance.
 * @param config A pointer to the ov_genai_generation_config instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_vlm_pipeline_set_generation_config(ov_genai_vlm_pipeline* pipe,
                                                                                 ov_genai_generation_config* config);
