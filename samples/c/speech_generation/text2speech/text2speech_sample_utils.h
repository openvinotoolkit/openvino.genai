// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>
#include <stdio.h>

#include "openvino/c/ov_tensor.h"

#define CHECK_STATUS(return_status)                                                           \
    if (return_status != 0) {                                                                 \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", (int)return_status, __LINE__); \
        exit_status = EXIT_FAILURE;                                                           \
        goto err;                                                                             \
    }

/**
 * @brief Save audio waveform to WAV file
 * @param waveform_ptr Pointer to float audio samples
 * @param waveform_size Number of samples
 * @param file_path Output WAV file path
 */
void save_to_wav(const float* waveform_ptr, size_t waveform_size, const char* file_path);

/**
 * @brief Read speaker embedding from binary file
 * @param file_path Path to binary file containing 512 float32 values
 * @return ov_tensor_t* containing the speaker embedding, or NULL on error
 */
ov_tensor_t* read_speaker_embedding(const char* file_path);
