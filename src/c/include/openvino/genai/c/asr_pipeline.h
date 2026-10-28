// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for  ov::genai::ASRPipeline class.
 *
 * @note Unless stated otherwise, status-returning functions report INVALID_C_PARAM for NULL required
 * arguments and UNKNOW_EXCEPTION for C++ exceptions. Free functions accept NULL.
 *
 * @file asr_pipeline.h
 */

#pragma once
#include "asr_generation_config.h"
#include "perf_metrics.h"

/**
 * @struct ov_genai_asr_decoded_result_chunk
 * @brief type define ov_genai_asr_decoded_result_chunk from ov_genai_asr_decoded_result_chunk_opaque
 */
typedef struct ov_genai_asr_decoded_result_chunk_opaque ov_genai_asr_decoded_result_chunk;

/**
 * @struct ov_genai_asr_decoded_results
 * @brief type define ov_genai_asr_decoded_results from ov_genai_asr_decoded_results_opaque
 */
typedef struct ov_genai_asr_decoded_results_opaque ov_genai_asr_decoded_results;

/**
 * @struct ov_genai_asr_pipeline
 * @brief type define ov_genai_asr_pipeline from ov_genai_asr_pipeline_opaque
 */
typedef struct ov_genai_asr_pipeline_opaque ov_genai_asr_pipeline;

/**
 * @brief Create ASRDecodedResultChunk
 * @param chunk A pointer to a newly allocated handle. Free with ov_genai_asr_decoded_result_chunk_free.
 * The handle owns a copy and remains valid independently of its source.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_result_chunk_create(ov_genai_asr_decoded_result_chunk** chunk);

/**
 * @brief Release the memory allocated by ov_genai_asr_decoded_result_chunk.
 * @param chunk A pointer to the ov_genai_asr_decoded_result_chunk to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_asr_decoded_result_chunk_free(ov_genai_asr_decoded_result_chunk* chunk);

/**
 * @brief Get start timestamp from ov_genai_asr_decoded_result_chunk.
 * @param chunk A pointer to the ov_genai_asr_decoded_result_chunk instance.
 * @param start_ts A pointer to the start timestamp value.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_result_chunk_get_start_ts(const ov_genai_asr_decoded_result_chunk* chunk, float* start_ts);

/**
 * @brief Get end timestamp from ov_genai_asr_decoded_result_chunk.
 * @param chunk A pointer to the ov_genai_asr_decoded_result_chunk instance.
 * @param end_ts A pointer to the end timestamp value.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_result_chunk_get_end_ts(const ov_genai_asr_decoded_result_chunk* chunk, float* end_ts);

/**
 * @brief Get text from ov_genai_asr_decoded_result_chunk.
 * @param chunk A pointer to the ov_genai_asr_decoded_result_chunk instance.
 * @param text A pointer to the pre-allocated text buffer. It can be set to NULL, in which case the
 * *text_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire text.
 * @param text_size A Pointer to the size of the text from the chunk, including the null terminator. If
 * text is not NULL, *text_size should be greater than or equal to the text size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_result_chunk_get_text(const ov_genai_asr_decoded_result_chunk* chunk,
                                           char* text,
                                           size_t* text_size);

/**
 * @brief Create ASRDecodedResults
 * @param results A pointer to a newly allocated handle. Free with ov_genai_asr_decoded_results_free.
 * The handle owns a copy and remains valid independently of its source.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_asr_decoded_results_create(ov_genai_asr_decoded_results** results);

/**
 * @brief Release the memory allocated by ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_asr_decoded_results_free(ov_genai_asr_decoded_results* results);

/**
 * @brief Get performance metrics from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param metrics A pointer to independently allocated base performance metrics. Free with
 * ov_genai_asr_decoded_results_perf_metrics_free.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_perf_metrics(const ov_genai_asr_decoded_results* results,
                                              ov_genai_perf_metrics** metrics);

/**
 * @brief Get number of text results from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param count A pointer to the number of text results.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_texts_count(const ov_genai_asr_decoded_results* results, size_t* count);

/**
 * @brief Get text result at specific index from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param index The index of the text result to retrieve.
 * @param text A pointer to the pre-allocated text buffer. It can be set to NULL, in which case the
 * *text_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire text.
 * @param text_size A Pointer to the size of the text from the results, including the null terminator. If
 * text is not NULL, *text_size should be greater than or equal to the text size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_text_at(const ov_genai_asr_decoded_results* results,
                                         size_t index,
                                         char* text,
                                         size_t* text_size);

/**
 * @brief Get score at specific index from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param index The index of the score to retrieve.
 * @param score A pointer to the score value.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_score_at(const ov_genai_asr_decoded_results* results, size_t index, float* score);

/**
 * @brief Check if chunks are available from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param has_chunks A pointer to the boolean indicating if chunks are available.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_has_chunks(const ov_genai_asr_decoded_results* results, bool* has_chunks);

/**
 * @brief Get number of chunks (zero if timestamps are unavailable) from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param result_index Transcription index in the nested timestamp arrays (zero for single audio input).
 * @param count A pointer to the number of chunks.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_chunks_count(const ov_genai_asr_decoded_results* results,
                                              size_t result_index,
                                              size_t* count);

/**
 * @brief Get chunk at specific index from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param result_index Transcription index in the nested timestamp arrays (zero for single audio input).
 * @param index The index of the chunk to retrieve.
 * @param chunk A pointer to a newly allocated handle. Free with ov_genai_asr_decoded_result_chunk_free.
 * The handle owns a copy and remains valid independently of its source.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_chunk_at(const ov_genai_asr_decoded_results* results,
                                          size_t result_index,
                                          size_t index,
                                          ov_genai_asr_decoded_result_chunk** chunk);

/**
 * @brief Get string representation from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param output A pointer to the pre-allocated output string buffer. It can be set to NULL, in which case the
 * *output_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire output.
 * @param output_size A Pointer to the size of the output string from the results, including the null terminator. If
 * output is not NULL, *output_size should be greater than or equal to the result string size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_string(const ov_genai_asr_decoded_results* results, char* output, size_t* output_size);

/**
 * @brief Construct ov_genai_asr_pipeline.
 *
 * Initializes a ov_genai_asr_pipeline instance from the specified model directory and device. Optional property
 * parameters can be passed as key-value pairs.
 *
 * @param models_path Path to the directory containing the model files.
 * @param device Name of a device to load a model to.
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param pipeline A pointer to a newly allocated handle. Free with ov_genai_asr_pipeline_free.
 * @param ... property parameter: Optional pack of pairs: <char* property_key, char* property_value> relevant only
 * @return ov_status_e A status code, return OK(0) if successful.
 *
 * @example
 * Example with no properties:
 * ov_genai_asr_pipeline_create(model_path, "CPU", 0, &pipeline);
 *
 * Example with properties:
 * ov_genai_asr_pipeline_create(model_path, "GPU", 2, &pipeline,
 *                                  "CACHE_DIR", "cache_dir");
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_asr_pipeline_create(const char* models_path,
                                                                  const char* device,
                                                                  const size_t property_args_size,
                                                                  ov_genai_asr_pipeline** pipeline,
                                                                  ...);

/**
 * @brief Release the memory allocated by ov_genai_asr_pipeline.
 * @param pipeline A pointer to the ov_genai_asr_pipeline to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_asr_pipeline_free(ov_genai_asr_pipeline* pipeline);

/**
 * @brief Generate results by ov_genai_asr_pipeline from raw speech input.
 * @param pipeline A pointer to the ov_genai_asr_pipeline instance.
 * @param raw_speech A pointer to the raw speech input array (float values).
 * @param raw_speech_size Number of normalized mono float samples at 16 kHz.
 * @param config A pointer to the ov_genai_asr_generation_config, NULL uses the pipeline's current configuration.
 * @param results A pointer to a newly allocated result handle. Free with ov_genai_asr_decoded_results_free.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_asr_pipeline_generate(ov_genai_asr_pipeline* pipeline,
                                                                    const float* raw_speech,
                                                                    size_t raw_speech_size,
                                                                    const ov_genai_asr_generation_config* config,
                                                                    ov_genai_asr_decoded_results** results);

/**
 * @brief Get the ASRGenerationConfig from ov_genai_asr_pipeline.
 * @param pipeline A pointer to the ov_genai_asr_pipeline instance.
 * @param config A pointer to a newly allocated handle. Free with ov_genai_asr_generation_config_free.
 * The handle owns a copy and remains valid independently of its source.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_pipeline_get_generation_config(const ov_genai_asr_pipeline* pipeline,
                                            ov_genai_asr_generation_config** config);

/**
 * @brief Set the ASRGenerationConfig to ov_genai_asr_pipeline.
 * @param pipeline A pointer to the ov_genai_asr_pipeline instance.
 * @param config A pointer to the ov_genai_asr_generation_config instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_pipeline_set_generation_config(ov_genai_asr_pipeline* pipeline, ov_genai_asr_generation_config* config);

/**
 * @brief Check if words are available from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param has_words A pointer to the boolean indicating if words are available.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_asr_decoded_results_has_words(const ov_genai_asr_decoded_results* results,
                                                                            bool* has_words);

/**
 * @brief Get number of words (zero if timestamps are unavailable) from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param result_index Transcription index in the nested timestamp arrays (zero for single audio input).
 * @param count A pointer to the number of words.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_words_count(const ov_genai_asr_decoded_results* results,
                                             size_t result_index,
                                             size_t* count);

/**
 * @brief Get word at specific index from ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param result_index Transcription index in the nested timestamp arrays (zero for single audio input).
 * @param index The index of the word to retrieve.
 * @param chunk A pointer to a newly allocated handle. Free with ov_genai_asr_decoded_result_chunk_free.
 * The handle owns a copy and remains valid independently of its source.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_word_at(const ov_genai_asr_decoded_results* results,
                                         size_t result_index,
                                         size_t index,
                                         ov_genai_asr_decoded_result_chunk** chunk);

/**
 * @brief Get detected or requested language (empty when unavailable) at a specific index from
 * ov_genai_asr_decoded_results.
 * @param results A pointer to the ov_genai_asr_decoded_results instance.
 * @param index The index of the language result to retrieve.
 * @param language A pointer to the pre-allocated language buffer. It can be set to NULL, in which case the
 * *language_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire language.
 * @param language_size A Pointer to the size of the language from the results, including the null terminator. If
 * language is not NULL, *language_size should be greater than or equal to the language size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_asr_decoded_results_get_language_at(const ov_genai_asr_decoded_results* results,
                                             size_t index,
                                             char* language,
                                             size_t* language_size);

/**
 * @brief Release performance metrics returned by the ASR results accessor.
 * @param metrics An owned metrics handle, or NULL.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_asr_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics);
