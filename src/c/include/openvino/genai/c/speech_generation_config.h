/**
 * @brief C API for speech generation configuration
 * @file speech_generation_config.h
 */

#pragma once

#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle for speech generation configuration
 */
typedef struct speech_generation_config_t* speech_generation_config_handle_t;

/**
 * @brief Creates a speech generation configuration
 * @param[out] config Pointer to store the created configuration handle
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_config_create(speech_generation_config_handle_t* config);

/**
 * @brief Creates a speech generation configuration from JSON file
 * @param[out] config Pointer to store the created configuration handle
 * @param[in] json_path Path to the JSON configuration file
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_config_create_from_json(speech_generation_config_handle_t* config,
                                                                  const char* json_path);

/**
 * @brief Destroys a speech generation configuration
 * @param[in] config Configuration handle to destroy
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_config_destroy(speech_generation_config_handle_t config);

/**
 * @brief Sets the minimum length ratio
 * @param[in] config Configuration handle
 * @param[in] value Minimum ratio of output length to input text length
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_config_set_minlenratio(speech_generation_config_handle_t config, float value);

/**
 * @brief Gets the minimum length ratio
 * @param[in] config Configuration handle
 * @param[out] value Pointer to store the minimum length ratio
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_config_get_minlenratio(speech_generation_config_handle_t config, float* value);

/**
 * @brief Sets the maximum length ratio
 * @param[in] config Configuration handle
 * @param[in] value Maximum ratio of output length to input text length
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_config_set_maxlenratio(speech_generation_config_handle_t config, float value);

/**
 * @brief Gets the maximum length ratio
 * @param[in] config Configuration handle
 * @param[out] value Pointer to store the maximum length ratio
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_config_get_maxlenratio(speech_generation_config_handle_t config, float* value);

/**
 * @brief Sets the probability threshold
 * @param[in] config Configuration handle
 * @param[in] value Probability threshold for stopping decoding
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_config_set_threshold(speech_generation_config_handle_t config, float value);

/**
 * @brief Gets the probability threshold
 * @param[in] config Configuration handle
 * @param[out] value Pointer to store the probability threshold
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_config_get_threshold(speech_generation_config_handle_t config, float* value);

#ifdef __cplusplus
}  // extern "C"
#endif
