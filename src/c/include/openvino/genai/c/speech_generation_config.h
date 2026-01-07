// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for ov::genai::SpeechGenerationConfig
 * class.
 *
 * @file speech_generation_config.h
 */

#pragma once

#include "openvino/genai/c/generation_config.h"

/**
 * @struct ov_genai_speech_generation_config
 * @brief type define ov_genai_speech_generation_config from ov_genai_speech_generation_config_opaque
 */
typedef struct ov_genai_speech_generation_config_opaque ov_genai_speech_generation_config;

/**
 * @brief Create SpeechGenerationConfig
 * @param config A pointer to the newly created ov_genai_speech_generation_config.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_create(ov_genai_speech_generation_config** config);

/**
 * @brief Create SpeechGenerationConfig from JSON file.
 * @param json_path Path to a .json file containing the generation configuration to load.
 * @param config A pointer to the newly created ov_genai_speech_generation_config.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_create_from_json(const char* json_path, ov_genai_speech_generation_config** config);

/**
 * @brief Release the memory allocated by ov_genai_speech_generation_config.
 * @param config A pointer to the ov_genai_speech_generation_config to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_speech_generation_config_free(ov_genai_speech_generation_config* config);

/**
 * @brief Set min length ratio for SpeechGenerationConfig.
 * @param config A pointer to the ov_genai_speech_generation_config instance.
 * @param minlenratio Minimum ratio of output length to input text length.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_minlenratio(ov_genai_speech_generation_config* config, float minlenratio);

/**
 * @brief Get min length ratio from SpeechGenerationConfig.
 * @param config A pointer to the ov_genai_speech_generation_config instance.
 * @param minlenratio A pointer to the minimum ratio of output length to input text length.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_minlenratio(const ov_genai_speech_generation_config* config, float* minlenratio);

/**
 * @brief Set max length ratio for SpeechGenerationConfig.
 * @param config A pointer to the ov_genai_speech_generation_config instance.
 * @param maxlenratio Maximum ratio of output length to input text length.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_maxlenratio(ov_genai_speech_generation_config* config, float maxlenratio);

/**
 * @brief Get max length ratio from SpeechGenerationConfig.
 * @param config A pointer to the ov_genai_speech_generation_config instance.
 * @param maxlenratio A pointer to the maximum ratio of output length to input text length.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_maxlenratio(const ov_genai_speech_generation_config* config, float* maxlenratio);

/**
 * @brief Set threshold for SpeechGenerationConfig.
 * @param config A pointer to the ov_genai_speech_generation_config instance.
 * @param threshold Probability threshold for stopping decoding.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_set_threshold(ov_genai_speech_generation_config* config, float threshold);

/**
 * @brief Get threshold from SpeechGenerationConfig.
 * @param config A pointer to the ov_genai_speech_generation_config instance.
 * @param threshold A pointer to the probability threshold for stopping decoding.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_speech_generation_config_get_threshold(const ov_genai_speech_generation_config* config, float* threshold);
