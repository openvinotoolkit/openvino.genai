// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/core/except.hpp"

// pack_latents: (B, C, H, W) -> (B, (H/2)*(W/2), C*4)
// Patchifies 2x2 spatial blocks into channels and transposes to sequence form.
inline ov::Tensor pack_latents(const ov::Tensor latents, const size_t batch_size, const size_t num_channels_latents, const size_t height, const size_t width) {
    size_t h_half = height / 2, w_half = width / 2;

    ov::Shape final_shape = {batch_size, h_half * w_half, num_channels_latents * 4};
    ov::Tensor permuted_latents = ov::Tensor(latents.get_element_type(), final_shape);

    OPENVINO_ASSERT(latents.get_size() == permuted_latents.get_size(), "Incorrect target shape, tensors must have the same sizes");

    auto src_data = latents.data<float>();
    float* dst_data = permuted_latents.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < num_channels_latents; ++c) {
            for (size_t h2 = 0; h2 < h_half; ++h2) {
                for (size_t w2 = 0; w2 < w_half; ++w2) {
                    size_t base_src_index = (b * num_channels_latents + c) * height * width + (h2 * 2 * width + w2 * 2);
                    size_t base_dst_index = (b * h_half * w_half + h2 * w_half + w2) * num_channels_latents * 4 + c * 4;

                    dst_data[base_dst_index] = src_data[base_src_index];
                    dst_data[base_dst_index + 1] = src_data[base_src_index + 1];
                    dst_data[base_dst_index + 2] = src_data[base_src_index + width];
                    dst_data[base_dst_index + 3] = src_data[base_src_index + width + 1];
                }
            }
        }
    }

    return permuted_latents;
}

// unpack_latents: (B, seq_len, C) -> (B, C/4, H, W)
// Inverse of pack_latents: un-transposes from sequence form and reconstructs 2x2 spatial patches.
inline ov::Tensor unpack_latents(const ov::Tensor& latents, const size_t height, const size_t width, const size_t vae_scale_factor) {
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape[0], channels = latents_shape[2];

    size_t out_height = height / vae_scale_factor;
    size_t out_width = width / vae_scale_factor;

    size_t h_half = out_height / 2;
    size_t w_half = out_width / 2;
    size_t c_quarter = channels / 4;

    ov::Shape final_shape = {batch_size, c_quarter, out_height, out_width};
    ov::Tensor permuted_latents(latents.get_element_type(), final_shape);

    OPENVINO_ASSERT(latents.get_size() == permuted_latents.get_size(), "Incorrect target shape, tensors must have the same sizes");

    const float* src_data = latents.data<float>();
    float* dst_data = permuted_latents.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c4 = 0; c4 < c_quarter; ++c4) {
            for (size_t h2 = 0; h2 < h_half; ++h2) {
                for (size_t w2 = 0; w2 < w_half; ++w2) {
                    size_t base_reshaped_index = (((b * h_half + h2) * w_half + w2) * c_quarter + c4) * 4;
                    size_t base_final_index = (b * c_quarter * out_height * out_width) + (c4 * out_height * out_width) + (h2 * 2 * out_width + w2 * 2);

                    dst_data[base_final_index] = src_data[base_reshaped_index];
                    dst_data[base_final_index + 1] = src_data[base_reshaped_index + 1];
                    dst_data[base_final_index + out_width] = src_data[base_reshaped_index + 2];
                    dst_data[base_final_index + out_width + 1] = src_data[base_reshaped_index + 3];
                }
            }
        }
    }

    return permuted_latents;
}
