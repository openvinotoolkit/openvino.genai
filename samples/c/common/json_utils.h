// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef SAMPLES_C_COMMON_JSON_UTILS_H_
#define SAMPLES_C_COMMON_JSON_UTILS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Escapes a string for use within a JSON value.
 *
 * Handles special characters (", \, control characters) and
 * preserves valid UTF-8 multi-byte sequences.
 *
 * @param input       Null-terminated input string.
 * @param output      Buffer to receive the escaped string.
 * @param output_size Size of the output buffer in bytes.
 * @return 0 on success, -1 if the buffer is too small or arguments are invalid.
 */
int json_escape_string(const char* input, char* output, size_t output_size);

#ifdef __cplusplus
}
#endif

#endif  // SAMPLES_C_COMMON_JSON_UTILS_H_
