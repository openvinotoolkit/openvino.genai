// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "openvino/c/ov_common.h"
#include "openvino/genai/c/visibility.h"

/** @brief Configuration for SpeechT5 and Kokoro text-to-speech generation. */
typedef struct ov_genai_speech_generation_config_opaque ov_genai_speech_generation_config;

/**
 * @brief Create a speech generation configuration with default values.
 * @param[out] config Receives a new configuration. Release it with ov_genai_speech_generation_config_free.
 * @return OK on success, INVALID_C_PARAM if config is NULL, or UNKNOW_EXCEPTION on construction failure.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_create(ov_genai_speech_generation_config** config);

/**
 * @brief Load a speech generation configuration from a JSON file.
 * @param[in] json_path Path to the configuration JSON file.
 * @param[out] config Receives a new configuration. Release it with ov_genai_speech_generation_config_free.
 * @return OK on success, INVALID_C_PARAM for NULL arguments, or UNKNOW_EXCEPTION if loading fails.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_create_from_json(const char* json_path, ov_genai_speech_generation_config** config);

/**
 * @brief Release a configuration created by this API. NULL is allowed.
 * @param[in] config Configuration to release.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_speech_generation_config_free(ov_genai_speech_generation_config* config);

/**
 * @brief Check the configuration for conflicting or invalid parameters.
 * @param[in] config Configuration to validate.
 * @return OK if valid, INVALID_C_PARAM if config is NULL, or UNKNOW_EXCEPTION if validation fails.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_validate(const ov_genai_speech_generation_config* config);

/**
 * @brief Set the minimum SpeechT5 output length ratio (default 0.0).
 * @param[in,out] config Configuration to update.
 * @param[in] value Minimum output length relative to input text length.
 * @return OK on success or INVALID_C_PARAM if config is NULL.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_minlenratio(ov_genai_speech_generation_config* config, float value);
/**
 * @brief Get the minimum SpeechT5 output length ratio.
 * @param[in] config Configuration to read.
 * @param[out] value Receives the ratio.
 * @return OK on success or INVALID_C_PARAM for NULL arguments.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_minlenratio(const ov_genai_speech_generation_config* config, float* value);
/**
 * @brief Set the maximum SpeechT5 output length ratio (default 20.0).
 * @param[in,out] config Configuration to update.
 * @param[in] value Maximum output length relative to input text length.
 * @return OK on success or INVALID_C_PARAM if config is NULL.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_maxlenratio(ov_genai_speech_generation_config* config, float value);
/**
 * @brief Get the maximum SpeechT5 output length ratio.
 * @param[in] config Configuration to read.
 * @param[out] value Receives the ratio.
 * @return OK on success or INVALID_C_PARAM for NULL arguments.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_maxlenratio(const ov_genai_speech_generation_config* config, float* value);
/**
 * @brief Set the SpeechT5 decoding stop threshold (default 0.5).
 * @param[in,out] config Configuration to update.
 * @param[in] value Stop threshold.
 * @return OK on success or INVALID_C_PARAM if config is NULL.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_threshold(ov_genai_speech_generation_config* config, float value);
/**
 * @brief Get the SpeechT5 decoding stop threshold.
 * @param[in] config Configuration to read.
 * @param[out] value Receives the threshold.
 * @return OK on success or INVALID_C_PARAM for NULL arguments.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_threshold(const ov_genai_speech_generation_config* config, float* value);
/**
 * @brief Set the Kokoro speech speed multiplier (default 1.0).
 * @param[in,out] config Configuration to update.
 * @param[in] value Speech speed multiplier.
 * @return OK on success or INVALID_C_PARAM if config is NULL.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_speed(ov_genai_speech_generation_config* config, float value);
/**
 * @brief Get the Kokoro speech speed multiplier.
 * @param[in] config Configuration to read.
 * @param[out] value Receives the multiplier.
 * @return OK on success or INVALID_C_PARAM for NULL arguments.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_speed(const ov_genai_speech_generation_config* config, float* value);
/**
 * @brief Set the maximum Kokoro phoneme sequence length per preprocessing chunk (default 510).
 * @param[in,out] config Configuration to update.
 * @param[in] value Maximum phoneme length.
 * @return OK on success or INVALID_C_PARAM if config is NULL.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_max_phoneme_length(ov_genai_speech_generation_config* config, uint32_t value);
/**
 * @brief Get the maximum Kokoro phoneme sequence length per preprocessing chunk.
 * @param[in] config Configuration to read.
 * @param[out] value Receives the maximum length.
 * @return OK on success or INVALID_C_PARAM for NULL arguments.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_max_phoneme_length(const ov_genai_speech_generation_config* config,
                                                         uint32_t* value);

/**
 * @brief Set the optional OpenVINO phonemizer fallback model directory used by Kokoro.
 * @param[in,out] config Configuration to update.
 * @param[in] path Directory path, or NULL to clear the fallback and use the default espeak-ng fallback.
 * @return OK on success, INVALID_C_PARAM if config is NULL, or UNKNOW_EXCEPTION on path conversion failure.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_phonemize_fallback_model_dir(ov_genai_speech_generation_config* config,
                                                                   const char* path);
/**
 * @brief Read the optional OpenVINO phonemizer fallback model directory.
 * @param[in] config Configuration to read.
 * @param[out] buffer Caller-owned UTF-8 buffer, or NULL to query the required byte count.
 * @param[in,out] size Buffer capacity on input; on success receives the required byte count, including the NUL
 *                     terminator. Query with buffer=NULL before allocating.
 * @return OK on success, NOT_FOUND if unset, OUT_OF_BOUNDS if the buffer is too small,
 *         INVALID_C_PARAM for NULL config or size, or UNKNOW_EXCEPTION on path conversion failure.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_phonemize_fallback_model_dir(const ov_genai_speech_generation_config* config,
                                                                   char* buffer,
                                                                   size_t* size);
/**
 * @brief Set the Kokoro G2P language code (default "en-us").
 * @param[in,out] config Configuration to update.
 * @param[in] language NUL-terminated language code, such as "en-us" or "en-gb".
 * @return OK on success, INVALID_C_PARAM for NULL arguments, or UNKNOW_EXCEPTION on allocation failure.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_language(ov_genai_speech_generation_config* config, const char* language);
/**
 * @brief Read the Kokoro G2P language code.
 * @param[in] config Configuration to read.
 * @param[out] buffer Caller-owned UTF-8 buffer, or NULL to query the required byte count.
 * @param[in,out] size Buffer capacity on input; on success receives the required byte count, including the NUL
 *                     terminator. Query with buffer=NULL before allocating.
 * @return OK on success, OUT_OF_BOUNDS if the buffer is too small, or INVALID_C_PARAM for NULL config or size.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_language(const ov_genai_speech_generation_config* config,
                                               char* buffer,
                                               size_t* size);
