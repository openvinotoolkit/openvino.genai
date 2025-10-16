// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "openvino/runtime/tensor.hpp"

// TODO: switch image normalization to double: CVS-174746
struct clip_ctx {
    float image_mean[3] = {0.0f, 0.0f, 0.0f};
    float image_std[3] = {1.0f, 1.0f, 1.0f};
    size_t image_size = 0;
};

struct clip_ctx_double {
    double image_mean[3] = {0.0, 0.0, 0.0};
    double image_std[3] = {1.0, 1.0, 1.0};
};

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

/**
 * @brief Converts an OpenVINO image tensor (1HWC) to a clip_image_u8 structure.
 *
 * @param image_tensor An OpenVINO tensor (1HWC) containing the image data.
 * @return A clip_image_u8 structure containing the image data.
 */
clip_image_u8 tensor_to_clip_image_u8(const ov::Tensor& image_tensor);

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

/**
 * @brief Converts a clip_image_f32 structure to an OpenVINO image tensor (1CHW).
 *
 * @param image A clip_image_f32 structure containing the image data.
 * @return An OpenVINO tensor containing the image data (1CHW).
 */
ov::Tensor clip_image_f32_to_tensor(const clip_image_f32& image);

void bicubic_resize(const clip_image_u8& img, clip_image_u8& dst, int target_width, int target_height);
void bilinear_resize(const clip_image_u8& src, clip_image_u8& dst, int target_width, int target_height);

/** preprocess img and store the result in res_imgs, pad_to_square may be overridden to false depending on model configuration */
clip_image_f32 clip_image_preprocess(struct clip_ctx& ctx, const clip_image_u8& img);

std::vector<clip_image_u8> get_image_patches(
    const clip_image_u8& image, 
    const std::vector<std::pair<int, int>>& image_grid_pinpoints,
    const std::pair<int, int>& size,
    int patch_size
);

std::pair<int, int> select_best_resolution(const std::pair<int, int> & original_size, const std::vector<std::pair<int, int>> & possible_resolutions);

clip_image_u8 resize_and_pad_image(const clip_image_u8& image, const std::pair<int, int>& target_resolution, uint8_t pad_value = 0);

clip_image_u8 center_crop(const clip_image_u8& image, size_t crop_height, size_t crop_width);

clip_image_f32 normalize_and_convert_to_chw(const clip_image_u8& img, const clip_ctx_double& image_mean_std);