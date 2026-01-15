// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include "openvino/runtime/tensor.hpp"

 /**
 * @brief Writes multiple videos (depending on `video` tensor batch size) to AVI file(s)
 * @param filename Output filename. File name or pattern to use to write video.
 * @param video_tensor Video(s) tensor of shape [B, F, H, W, C] with uint8 data (C = 1, 3, or 4).
 * @param fps Frames per second.
 * @param input_is_rgb True if the input frames are in RGB/RGBA format (will be converted to BGR for OpenCV).
 */
void save_video(const std::string& filename, const ov::Tensor& video_tensor, int fps = 25, bool input_is_rgb = true);
