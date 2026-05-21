// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for ov::genai::TextEmbeddingPipeline
 * class.
 *
 * @file text_embedding_pipeline.h
 */

#pragma once
#include "openvino/c/ov_common.h"
#include "openvino/genai/c/visibility.h"

/**
 * @enum ov_genai_embedding_dtype_e
 * @brief Element type of an embedding tensor.
 */
typedef enum {
    OV_GENAI_EMBEDDING_DTYPE_F32 = 0,  ///< IEEE 754 binary32 (float)
    OV_GENAI_EMBEDDING_DTYPE_I8 = 1,   ///< signed 8-bit integer
    OV_GENAI_EMBEDDING_DTYPE_U8 = 2,   ///< unsigned 8-bit integer
} ov_genai_embedding_dtype_e;

/**
 * @struct ov_genai_embedding_result
 * @brief type define ov_genai_embedding_result from ov_genai_embedding_result_opaque
 *
 * Holds a single embedding vector returned by embed_query.
 */
typedef struct ov_genai_embedding_result_opaque ov_genai_embedding_result;

/**
 * @brief Release the memory allocated by ov_genai_embedding_result.
 * @param result A pointer to the ov_genai_embedding_result to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_embedding_result_free(ov_genai_embedding_result* result);

/**
 * @brief Get the element type of the embedding.
 * @param result A pointer to the ov_genai_embedding_result instance.
 * @param dtype A pointer to ov_genai_embedding_dtype_e which receives the element type.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_result_get_dtype(const ov_genai_embedding_result* result,
                                                                         ov_genai_embedding_dtype_e* dtype);

/**
 * @brief Get the number of elements (vector length) in the embedding.
 * @param result A pointer to the ov_genai_embedding_result instance.
 * @param size A pointer to size_t which receives the number of elements.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_result_get_size(const ov_genai_embedding_result* result,
                                                                        size_t* size);

/**
 * @brief Borrow the raw float32 buffer of the embedding.
 *
 * The returned pointer is owned by the result object and is invalidated when the result is freed.
 *
 * @param result A pointer to the ov_genai_embedding_result instance.
 * @param data Receives a pointer to the underlying float buffer.
 * @param size Receives the number of float elements.
 * @return ov_status_e A status code, return OK(0) if successful. Returns INVALID_C_PARAM(-2) if the embedding dtype is
 * not F32.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_result_get_data_f32(const ov_genai_embedding_result* result,
                                                                            const float** data,
                                                                            size_t* size);

/**
 * @brief Borrow the raw int8 buffer of the embedding.
 *
 * The returned pointer is owned by the result object and is invalidated when the result is freed.
 *
 * @param result A pointer to the ov_genai_embedding_result instance.
 * @param data Receives a pointer to the underlying int8 buffer.
 * @param size Receives the number of int8 elements.
 * @return ov_status_e A status code, return OK(0) if successful. Returns INVALID_C_PARAM(-2) if the embedding dtype is
 * not I8.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_result_get_data_i8(const ov_genai_embedding_result* result,
                                                                           const int8_t** data,
                                                                           size_t* size);

/**
 * @brief Borrow the raw uint8 buffer of the embedding.
 *
 * The returned pointer is owned by the result object and is invalidated when the result is freed.
 *
 * @param result A pointer to the ov_genai_embedding_result instance.
 * @param data Receives a pointer to the underlying uint8 buffer.
 * @param size Receives the number of uint8 elements.
 * @return ov_status_e A status code, return OK(0) if successful. Returns INVALID_C_PARAM(-2) if the embedding dtype is
 * not U8.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_result_get_data_u8(const ov_genai_embedding_result* result,
                                                                           const uint8_t** data,
                                                                           size_t* size);

/**
 * @struct ov_genai_embedding_results
 * @brief type define ov_genai_embedding_results from ov_genai_embedding_results_opaque
 *
 * Holds a batch of embedding vectors returned by embed_documents. All inner vectors share the same dtype.
 */
typedef struct ov_genai_embedding_results_opaque ov_genai_embedding_results;

/**
 * @brief Release the memory allocated by ov_genai_embedding_results.
 * @param results A pointer to the ov_genai_embedding_results to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_embedding_results_free(ov_genai_embedding_results* results);

/**
 * @brief Get the element type of the batch (shared by all inner vectors).
 * @param results A pointer to the ov_genai_embedding_results instance.
 * @param dtype A pointer to ov_genai_embedding_dtype_e which receives the element type.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_results_get_dtype(const ov_genai_embedding_results* results,
                                                                          ov_genai_embedding_dtype_e* dtype);

/**
 * @brief Get the number of embedding vectors in the batch.
 * @param results A pointer to the ov_genai_embedding_results instance.
 * @param count A pointer to size_t which receives the number of embeddings.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_results_get_count(const ov_genai_embedding_results* results,
                                                                          size_t* count);

/**
 * @brief Get the number of elements in the i-th embedding vector.
 * @param results A pointer to the ov_genai_embedding_results instance.
 * @param i Index of the inner vector. Must be less than the value returned by ov_genai_embedding_results_get_count().
 * @param size A pointer to size_t which receives the number of elements.
 * @return ov_status_e A status code, return OK(0) if successful. Returns OUT_OF_BOUNDS(-6) if i is out of range.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_results_get_size_at(const ov_genai_embedding_results* results,
                                                                            size_t i,
                                                                            size_t* size);

/**
 * @brief Borrow the raw float32 buffer of the i-th embedding vector.
 *
 * The returned pointer is owned by the results object and is invalidated when the results are freed.
 *
 * @param results A pointer to the ov_genai_embedding_results instance.
 * @param i Index of the inner vector.
 * @param data Receives a pointer to the float buffer.
 * @param size Receives the number of float elements.
 * @return ov_status_e A status code, return OK(0) if successful. Returns INVALID_C_PARAM(-2) if dtype is not F32, or
 * OUT_OF_BOUNDS(-6) if i is out of range.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_results_get_data_f32_at(
    const ov_genai_embedding_results* results,
    size_t i,
    const float** data,
    size_t* size);

/**
 * @brief Borrow the raw int8 buffer of the i-th embedding vector.
 *
 * The returned pointer is owned by the results object and is invalidated when the results are freed.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_results_get_data_i8_at(
    const ov_genai_embedding_results* results,
    size_t i,
    const int8_t** data,
    size_t* size);

/**
 * @brief Borrow the raw uint8 buffer of the i-th embedding vector.
 *
 * The returned pointer is owned by the results object and is invalidated when the results are freed.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_embedding_results_get_data_u8_at(
    const ov_genai_embedding_results* results,
    size_t i,
    const uint8_t** data,
    size_t* size);

/**
 * @struct ov_genai_text_embedding_pipeline
 * @brief type define ov_genai_text_embedding_pipeline from ov_genai_text_embedding_pipeline_opaque
 */
typedef struct ov_genai_text_embedding_pipeline_opaque ov_genai_text_embedding_pipeline;

/**
 * @brief Construct ov_genai_text_embedding_pipeline.
 *
 * Initializes a ov_genai_text_embedding_pipeline instance from the specified model directory and device. Optional
 * property parameters can be passed as key-value pairs.
 *
 * Recognized embedding-specific config keys (passed as strings, converted internally):
 *   - "max_length"         : size_t, maximum tokens passed to the model
 *   - "pad_to_max_length"  : "true" or "false"
 *   - "padding_side"       : "left" or "right"
 *   - "batch_size"         : size_t, fixed model batch (used for database population)
 *   - "pooling_type"       : "CLS", "MEAN" or "LAST_TOKEN"
 *   - "normalize"          : "true" or "false" (L2 normalization of embeddings)
 *   - "query_instruction"  : string, instruction prepended to query inputs
 *   - "embed_instruction"  : string, instruction prepended to document inputs
 * Any other key is treated as an OpenVINO plugin property and forwarded as a string.
 *
 * @param models_path Path to the directory containing the model files.
 * @param device Name of a device to load a model to.
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param pipe A pointer to the newly created ov_genai_text_embedding_pipeline.
 * @param ... Optional pack of pairs: <const char* property_key, const char* property_value>.
 * @return ov_status_e A status code, return OK(0) if successful.
 *
 * @example
 * ov_genai_text_embedding_pipeline_create(model_path, "CPU", 0, &pipe);
 * ov_genai_text_embedding_pipeline_create(model_path, "CPU", 2, &pipe, "pooling_type", "MEAN");
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text_embedding_pipeline_create(const char* models_path,
                                        const char* device,
                                        const size_t property_args_size,
                                        ov_genai_text_embedding_pipeline** pipe,
                                        ...);

/**
 * @brief Release the memory allocated by ov_genai_text_embedding_pipeline.
 * @param pipe A pointer to the ov_genai_text_embedding_pipeline to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text_embedding_pipeline_free(ov_genai_text_embedding_pipeline* pipe);

/**
 * @brief Compute embeddings for a list of documents.
 *
 * @param pipe A pointer to the ov_genai_text_embedding_pipeline instance.
 * @param texts An array of null-terminated document strings.
 * @param texts_count Number of elements in `texts`.
 * @param results A pointer to the newly created ov_genai_embedding_results.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text_embedding_pipeline_embed_documents(ov_genai_text_embedding_pipeline* pipe,
                                                 const char** texts,
                                                 size_t texts_count,
                                                 ov_genai_embedding_results** results);

/**
 * @brief Asynchronously compute embeddings for a list of documents.
 *
 * Only one async call can be active on the pipeline at a time. The caller must follow this with
 * ov_genai_text_embedding_pipeline_wait_embed_documents() to retrieve the result.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text_embedding_pipeline_start_embed_documents_async(ov_genai_text_embedding_pipeline* pipe,
                                                             const char** texts,
                                                             size_t texts_count);

/**
 * @brief Wait for the result of a previously started asynchronous embed_documents.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text_embedding_pipeline_wait_embed_documents(ov_genai_text_embedding_pipeline* pipe,
                                                      ov_genai_embedding_results** results);

/**
 * @brief Compute embedding for a single query.
 *
 * @param pipe A pointer to the ov_genai_text_embedding_pipeline instance.
 * @param text A null-terminated query string.
 * @param result A pointer to the newly created ov_genai_embedding_result.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text_embedding_pipeline_embed_query(ov_genai_text_embedding_pipeline* pipe,
                                             const char* text,
                                             ov_genai_embedding_result** result);

/**
 * @brief Asynchronously compute embedding for a single query.
 *
 * Only one async call can be active on the pipeline at a time. The caller must follow this with
 * ov_genai_text_embedding_pipeline_wait_embed_query() to retrieve the result.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text_embedding_pipeline_start_embed_query_async(ov_genai_text_embedding_pipeline* pipe, const char* text);

/**
 * @brief Wait for the result of a previously started asynchronous embed_query.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text_embedding_pipeline_wait_embed_query(ov_genai_text_embedding_pipeline* pipe,
                                                  ov_genai_embedding_result** result);
