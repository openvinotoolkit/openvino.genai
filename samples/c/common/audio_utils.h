// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

#include <stdint.h>
#include <stdlib.h>

#include "openvino/c/ov_tensor.h"

int load_wav_file(const char* filename, float** audio_data, size_t* audio_length, float* sample_rate);

float* resample_audio(const float* input,
                      size_t input_length,
                      float input_rate,
                      float target_rate,
                      size_t* output_length);

// Reads mono WAV and converts it to a rank-1 F32 tensor. Returns 0 if sample
// rate does not match expected_sample_rate.
int load_wav_as_tensor(const char* filename, float expected_sample_rate, ov_tensor_t** tensor);

// Saves float32 mono samples into a 32-bit IEEE-float WAV file.
int save_wav_file(const char* path, const float* samples, size_t count, uint32_t sample_rate);

#endif  // AUDIO_UTILS_H
