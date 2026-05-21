// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for ov::genai::TextRerankPipeline class.
 *
 * @file text_rerank_pipeline.h
 */

#pragma once
#include "openvino/c/ov_common.h"
#include "openvino/genai/c/visibility.h"

/**
 * @struct ov_genai_text_rerank_result
 * @brief type define ov_genai_text_rerank_result from ov_genai_text_rerank_result_opaque
 *
 * Holds the result of a rerank call: a list of pairs (original index, score) sorted by score in descending order.
 */
typedef struct ov_genai_text_rerank_result_opaque ov_genai_text_rerank_result;

/**
 * @brief Release the memory allocated by ov_genai_text_rerank_result.
 * @param result A pointer to the ov_genai_text_rerank_result to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text_rerank_result_free(ov_genai_text_rerank_result* result);

/**
 * @brief Get the number of (index, score) pairs in the rerank result.
 * @param result A pointer to the ov_genai_text_rerank_result instance.
 * @param size A pointer to size_t which receives the number of pairs.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text_rerank_result_get_size(const ov_genai_text_rerank_result* result,
                                                                          size_t* size);

/**
 * @brief Get the (index, score) pair at the given position.
 * @param result A pointer to the ov_genai_text_rerank_result instance.
 * @param i Position in the result list. Must be less than the value returned by ov_genai_text_rerank_result_get_size().
 * @param index A pointer to size_t which receives the original document index in the input array.
 * @param score A pointer to float which receives the relevance score.
 * @return ov_status_e A status code, return OK(0) if successful. Returns OUT_OF_BOUNDS(-6) if i is out of range.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text_rerank_result_get_item(const ov_genai_text_rerank_result* result,
                                                                          size_t i,
                                                                          size_t* index,
                                                                          float* score);

/**
 * @struct ov_genai_text_rerank_pipeline
 * @brief type define ov_genai_text_rerank_pipeline from ov_genai_text_rerank_pipeline_opaque
 */
typedef struct ov_genai_text_rerank_pipeline_opaque ov_genai_text_rerank_pipeline;

/**
 * @brief Construct ov_genai_text_rerank_pipeline.
 *
 * Initializes a ov_genai_text_rerank_pipeline instance from the specified model directory and device. Optional
 * property parameters can be passed as key-value pairs.
 *
 * Recognized rerank-specific config keys (passed as strings, converted internally):
 *   - "top_n"             : size_t, number of documents to return (default 3)
 *   - "max_length"        : size_t, maximum tokens passed to the model
 *   - "pad_to_max_length" : "true" or "false"
 *   - "padding_side"      : "left" or "right"
 * Any other key is treated as an OpenVINO plugin property and forwarded as a string.
 *
 * @param models_path Path to the directory containing the model files.
 * @param device Name of a device to load a model to.
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param pipe A pointer to the newly created ov_genai_text_rerank_pipeline.
 * @param ... Optional pack of pairs: <const char* property_key, const char* property_value>.
 * @return ov_status_e A status code, return OK(0) if successful.
 *
 * @example
 * ov_genai_text_rerank_pipeline_create(model_path, "CPU", 0, &pipe);
 * ov_genai_text_rerank_pipeline_create(model_path, "CPU", 2, &pipe, "top_n", "5");
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text_rerank_pipeline_create(const char* models_path,
                                                                          const char* device,
                                                                          const size_t property_args_size,
                                                                          ov_genai_text_rerank_pipeline** pipe,
                                                                          ...);

/**
 * @brief Release the memory allocated by ov_genai_text_rerank_pipeline.
 * @param pipe A pointer to the ov_genai_text_rerank_pipeline to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text_rerank_pipeline_free(ov_genai_text_rerank_pipeline* pipe);

/**
 * @brief Rerank a list of texts against the query.
 *
 * The result holds at most `top_n` (index, score) pairs sorted by score in descending order, where `index` refers to
 * the position of the document in the input `texts` array.
 *
 * @param pipe A pointer to the ov_genai_text_rerank_pipeline instance.
 * @param query A null-terminated query string.
 * @param texts An array of null-terminated document strings.
 * @param texts_count Number of elements in `texts`.
 * @param result A pointer to the newly created ov_genai_text_rerank_result.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text_rerank_pipeline_rerank(ov_genai_text_rerank_pipeline* pipe,
                                                                          const char* query,
                                                                          const char** texts,
                                                                          size_t texts_count,
                                                                          ov_genai_text_rerank_result** result);

/**
 * @brief Asynchronously rerank a list of texts against the query.
 *
 * Only one async call can be active at a time. The caller must follow this with
 * ov_genai_text_rerank_pipeline_wait_rerank() to retrieve the result.
 *
 * @param pipe A pointer to the ov_genai_text_rerank_pipeline instance.
 * @param query A null-terminated query string.
 * @param texts An array of null-terminated document strings.
 * @param texts_count Number of elements in `texts`.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text_rerank_pipeline_start_rerank_async(
    ov_genai_text_rerank_pipeline* pipe,
    const char* query,
    const char** texts,
    size_t texts_count);

/**
 * @brief Wait for the result of a previously started asynchronous rerank.
 * @param pipe A pointer to the ov_genai_text_rerank_pipeline instance.
 * @param result A pointer to the newly created ov_genai_text_rerank_result.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text_rerank_pipeline_wait_rerank(ov_genai_text_rerank_pipeline* pipe,
                                                                               ov_genai_text_rerank_result** result);
