// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include "openvino/runtime/tensor.hpp"

/**
 * @brief Writes audio track(s) to WAV file(s) as 32-bit float PCM.
 * @param filename Output filename. If batch size > 1, files are named with "_b{N}" suffix.
 * @param audio_tensor Audio tensor of shape [B, C, S] with float data in [-1, 1].
 * @param sample_rate Audio sample rate in Hz.
 */
void save_audio(const std::string& filename, const ov::Tensor& audio_tensor, uint32_t sample_rate);
