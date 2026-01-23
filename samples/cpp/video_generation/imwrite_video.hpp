// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include "openvino/runtime/tensor.hpp"

/**
 * @brief Writes video(s) to AVI file(s). Input frames are assumed to be in RGB/RGBA format.
 * @param filename Output filename. If batch size > 1, files are named with "_b{N}" suffix.
 * @param video_tensor Video tensor of shape [B, F, H, W, C] with uint8 data (C = 1, 3, or 4).
 * @param fps Frames per second.
 */
void save_video(const std::string& filename, const ov::Tensor& video_tensor, float fps = 25.0f);
