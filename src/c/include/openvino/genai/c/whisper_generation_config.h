// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for  ov::genai::WhisperGenerationConfig
 * class.
 *
 * @file whisper_generation_config.h
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "generation_config.h"
#include "openvino/c/ov_common.h"
#include "openvino/genai/c/visibility.h"

/**
 * @struct ov_genai_whisper_generation_config
 * @brief type define ov_genai_whisper_generation_config from ov_genai_whisper_generation_config_opaque
 */
typedef struct ov_genai_whisper_generation_config_opaque ov_genai_whisper_generation_config;

/**
 * @brief Create ov_genai_whisper_generation_config.
 * @param config A pointer to the newly created ov_genai_whisper_generation_config.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_create(ov_genai_whisper_generation_config** config);

/**
 * @brief Create ov_genai_whisper_generation_config from JSON file.
 * @param json_path Path to a .json file containing the whisper generation configuration to load.
 * @param config A pointer to the newly created ov_genai_whisper_generation_config.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_create_from_json(const char* json_path, ov_genai_whisper_generation_config** config);

/**
 * @brief Release the memory allocated by ov_genai_whisper_generation_config.
 * @param config A pointer to the ov_genai_whisper_generation_config to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_whisper_generation_config_free(ov_genai_whisper_generation_config* config);

/**
 * @brief Get the underlying GenerationConfig from ov_genai_whisper_generation_config.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param generation_config A pointer to the newly created ov_genai_generation_config.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_generation_config(const ov_genai_whisper_generation_config* config,
                                                         ov_genai_generation_config** generation_config);

/**
 * @brief Set the decoder start token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id The decoder start token id (default: 50258).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_decoder_start_token_id(ov_genai_whisper_generation_config* config,
                                                              int64_t token_id);

/**
 * @brief Get the decoder start token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id A pointer to the decoder start token id.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_decoder_start_token_id(const ov_genai_whisper_generation_config* config,
                                                              int64_t* token_id);

/**
 * @brief Set the padding token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id The padding token id (default: 50257).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_pad_token_id(ov_genai_whisper_generation_config* config, int64_t token_id);

/**
 * @brief Get the padding token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id A pointer to the padding token id.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_pad_token_id(const ov_genai_whisper_generation_config* config,
                                                    int64_t* token_id);

/**
 * @brief Set the translate token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id The translate token id (default: 50358).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_translate_token_id(ov_genai_whisper_generation_config* config, int64_t token_id);

/**
 * @brief Get the translate token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id A pointer to the translate token id.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_translate_token_id(const ov_genai_whisper_generation_config* config,
                                                          int64_t* token_id);

/**
 * @brief Set the transcribe token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id The transcribe token id (default: 50359).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_transcribe_token_id(ov_genai_whisper_generation_config* config,
                                                           int64_t token_id);

/**
 * @brief Get the transcribe token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id A pointer to the transcribe token id.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_transcribe_token_id(const ov_genai_whisper_generation_config* config,
                                                           int64_t* token_id);

/**
 * @brief Set the previous start of transcript token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id The previous start of transcript token id (default: 50361).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_prev_sot_token_id(ov_genai_whisper_generation_config* config, int64_t token_id);

/**
 * @brief Get the previous start of transcript token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id A pointer to the previous start of transcript token id.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_prev_sot_token_id(const ov_genai_whisper_generation_config* config,
                                                         int64_t* token_id);

/**
 * @brief Set the no timestamps token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id The no timestamps token id (default: 50363).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_no_timestamps_token_id(ov_genai_whisper_generation_config* config,
                                                              int64_t token_id);

/**
 * @brief Get the no timestamps token id.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param token_id A pointer to the no timestamps token id.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_no_timestamps_token_id(const ov_genai_whisper_generation_config* config,
                                                              int64_t* token_id);

/**
 * @brief Set the maximum initial timestamp index.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param index The maximum initial timestamp index (default: 50).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_max_initial_timestamp_index(ov_genai_whisper_generation_config* config,
                                                                   size_t index);

/**
 * @brief Get the maximum initial timestamp index.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param index A pointer to the maximum initial timestamp index.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_max_initial_timestamp_index(const ov_genai_whisper_generation_config* config,
                                                                   size_t* index);

/**
 * @brief Set whether the model is multilingual.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param is_multilingual True if the model is multilingual (default: true).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_is_multilingual(ov_genai_whisper_generation_config* config,
                                                       bool is_multilingual);

/**
 * @brief Get whether the model is multilingual.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param is_multilingual A pointer to the multilingual flag.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_is_multilingual(const ov_genai_whisper_generation_config* config,
                                                       bool* is_multilingual);

/**
 * @brief Set the language for generation.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param language The language token (e.g., "en", "fr", "de"). Can be NULL to unset.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_language(ov_genai_whisper_generation_config* config, const char* language);

/**
 * @brief Get the language for generation.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param language A pointer to the pre-allocated language buffer. It can be set to NULL, in which case the
 * *language_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire language.
 * @param language_size A Pointer to the size of the language buffer, including the null terminator. If
 * language is not NULL, *language_size should be greater than or equal to the language size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful. NOT_FOUND(-5) if language is not set.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_language(const ov_genai_whisper_generation_config* config,
                                                char* language,
                                                size_t* language_size);

/**
 * @brief Set the task for generation.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param task The task ("translate" or "transcribe"). Can be NULL to unset.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_task(ov_genai_whisper_generation_config* config, const char* task);

/**
 * @brief Get the task for generation.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param task A pointer to the pre-allocated task buffer. It can be set to NULL, in which case the
 * *task_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire task.
 * @param task_size A Pointer to the size of the task buffer, including the null terminator. If
 * task is not NULL, *task_size should be greater than or equal to the task size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful. NOT_FOUND(-5) if task is not set.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_task(const ov_genai_whisper_generation_config* config,
                                            char* task,
                                            size_t* task_size);

/**
 * @brief Set whether to return timestamps.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param return_timestamps True to return timestamps for segments (default: false).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_return_timestamps(ov_genai_whisper_generation_config* config,
                                                         bool return_timestamps);

/**
 * @brief Get whether to return timestamps.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param return_timestamps A pointer to the return timestamps flag.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_return_timestamps(const ov_genai_whisper_generation_config* config,
                                                         bool* return_timestamps);

/**
 * @brief Set the initial prompt for generation.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param initial_prompt The initial prompt text. Can be NULL to unset.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_initial_prompt(ov_genai_whisper_generation_config* config,
                                                      const char* initial_prompt);

/**
 * @brief Get the initial prompt for generation.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param initial_prompt A pointer to the pre-allocated initial prompt buffer. It can be set to NULL, in which case the
 * *prompt_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire initial prompt.
 * @param prompt_size A Pointer to the size of the initial prompt buffer, including the null terminator. If
 * initial_prompt is not NULL, *prompt_size should be greater than or equal to the prompt size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful. NOT_FOUND(-5) if initial prompt is not set.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_initial_prompt(const ov_genai_whisper_generation_config* config,
                                                      char* initial_prompt,
                                                      size_t* prompt_size);

/**
 * @brief Set the hotwords for generation.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param hotwords The hotwords text. Can be NULL to unset.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_hotwords(ov_genai_whisper_generation_config* config, const char* hotwords);

/**
 * @brief Get the hotwords for generation.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param hotwords A pointer to the pre-allocated hotwords buffer. It can be set to NULL, in which case the
 * *hotwords_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire hotwords.
 * @param hotwords_size A Pointer to the size of the hotwords buffer, including the null terminator. If
 * hotwords is not NULL, *hotwords_size should be greater than or equal to the hotwords size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful. NOT_FOUND(-5) if hotwords is not set.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_hotwords(const ov_genai_whisper_generation_config* config,
                                                char* hotwords,
                                                size_t* hotwords_size);

/**
 * @brief Set the begin suppress tokens.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param tokens A pointer to the array of token ids to suppress at the beginning.
 * @param tokens_count The number of tokens in the array.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_begin_suppress_tokens(ov_genai_whisper_generation_config* config,
                                                             const int64_t* tokens,
                                                             size_t tokens_count);

/**
 * @brief Get the begin suppress tokens count.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param tokens_count A pointer to the number of begin suppress tokens.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_begin_suppress_tokens_count(const ov_genai_whisper_generation_config* config,
                                                                   size_t* tokens_count);

/**
 * @brief Get the begin suppress tokens.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param tokens A pointer to the pre-allocated array of token ids. The array should be allocated with the size
 * returned by ov_genai_whisper_generation_config_get_begin_suppress_tokens_count.
 * @param tokens_count The size of the tokens array.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_begin_suppress_tokens(const ov_genai_whisper_generation_config* config,
                                                             int64_t* tokens,
                                                             size_t tokens_count);

/**
 * @brief Set the suppress tokens.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param tokens A pointer to the array of token ids to suppress during generation.
 * @param tokens_count The number of tokens in the array.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_set_suppress_tokens(ov_genai_whisper_generation_config* config,
                                                       const int64_t* tokens,
                                                       size_t tokens_count);

/**
 * @brief Get the suppress tokens count.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param tokens_count A pointer to the number of suppress tokens.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_suppress_tokens_count(const ov_genai_whisper_generation_config* config,
                                                             size_t* tokens_count);

/**
 * @brief Get the suppress tokens.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @param tokens A pointer to the pre-allocated array of token ids. The array should be allocated with the size
 * returned by ov_genai_whisper_generation_config_get_suppress_tokens_count.
 * @param tokens_count The size of the tokens array.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_get_suppress_tokens(const ov_genai_whisper_generation_config* config,
                                                       int64_t* tokens,
                                                       size_t tokens_count);

/**
 * @brief Validate the whisper generation configuration.
 * @param config A pointer to the ov_genai_whisper_generation_config instance.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_whisper_generation_config_validate(ov_genai_whisper_generation_config* config);
