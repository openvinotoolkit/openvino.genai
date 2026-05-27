// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// This is a C wrapper for ov::genai::JsonContainer class.

#pragma once

#include "visibility.h"
#include <stddef.h>

/**
 * @struct ov_genai_json_container
 * @brief Opaque type for JsonContainer
 */
typedef struct ov_genai_json_container_opaque ov_genai_json_container;

/**
 * @brief Status codes for JsonContainer operations
 */
typedef enum {
    OV_GENAI_JSON_CONTAINER_OK = 0,
    OV_GENAI_JSON_CONTAINER_INVALID_PARAM = -1,
    OV_GENAI_JSON_CONTAINER_INVALID_JSON = -2,
    OV_GENAI_JSON_CONTAINER_OUT_OF_BOUNDS = -3,
    OV_GENAI_JSON_CONTAINER_ERROR = -4
} ov_genai_json_container_status_e;

/**
 * @brief Create a new empty JsonContainer instance (as an object).
 * @param container A pointer to the newly created ov_genai_json_container.
 * @return ov_genai_json_container_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_json_container_status_e ov_genai_json_container_create(ov_genai_json_container** container);

/**
 * @brief Create a JsonContainer instance from a JSON string.
 * @param container A pointer to the newly created ov_genai_json_container.
 * @param json_str A JSON string (object, array, or primitive).
 * @return ov_genai_json_container_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_json_container_status_e ov_genai_json_container_create_from_json_string(
    ov_genai_json_container** container,
    const char* json_str);

/**
 * @brief Create a JsonContainer instance as an empty JSON object.
 * @param container A pointer to the newly created ov_genai_json_container.
 * @return ov_genai_json_container_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_json_container_status_e ov_genai_json_container_create_object(ov_genai_json_container** container);

/**
 * @brief Create a JsonContainer instance as an empty JSON array.
 * @param container A pointer to the newly created ov_genai_json_container.
 * @return ov_genai_json_container_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_json_container_status_e ov_genai_json_container_create_array(ov_genai_json_container** container);

/**
 * @brief Release the memory allocated by ov_genai_json_container.
 * @param container A pointer to the ov_genai_json_container to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_json_container_free(ov_genai_json_container* container);

/**
 * @brief Convert JsonContainer to JSON string.
 * @param container A pointer to the ov_genai_json_container instance.
 * @param output A pointer to the pre-allocated output string buffer. It can be set to NULL, in which case the
 * *output_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire output.
 * @param output_size A pointer to the size of the output string, including the null terminator. If output is not NULL,
 * *output_size should be greater than or equal to the result string size; otherwise, the function will return
 * OUT_OF_BOUNDS(-3).
 * @return ov_genai_json_container_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_json_container_status_e ov_genai_json_container_to_json_string(
    const ov_genai_json_container* container,
    char* output,
    size_t* output_size);

/**
 * @brief Create a copy of JsonContainer.
 * @param source A pointer to the source ov_genai_json_container instance.
 * @param target A pointer to store the copied ov_genai_json_container.
 * @return ov_genai_json_container_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_json_container_status_e ov_genai_json_container_copy(
    const ov_genai_json_container* source,
    ov_genai_json_container** target);
