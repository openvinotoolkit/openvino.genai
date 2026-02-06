// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/tensor.hpp"
#include <vector>
#include <cstdint>

namespace ov {
namespace genai {
namespace module {
namespace blend_utils {

/**
 * @brief Blend two 4D tensors along height dimension (vertical seam removal)
 *
 * For image tensors with layout [N, C, H, W] (NCHW, f32)
 * Blends the bottom blend_extent rows of tensor 'a' with top blend_extent rows of tensor 'b'
 * Result is written to tensor 'b'
 *
 * @param a First tensor (above)
 * @param b Second tensor (below) - modified in place
 * @param blend_extent Number of rows to blend
 */
void blend_v_4d(ov::Tensor& a, ov::Tensor& b, size_t blend_extent);

/**
 * @brief Blend two 4D tensors along width dimension (horizontal seam removal)
 *
 * For image tensors with layout [N, C, H, W] (NCHW, f32)
 * Blends the rightmost blend_extent columns of tensor 'a' with leftmost blend_extent columns of tensor 'b'
 * Result is written to tensor 'b'
 *
 * @param a First tensor (left)
 * @param b Second tensor (right) - modified in place
 * @param blend_extent Number of columns to blend
 */
void blend_h_4d(ov::Tensor& a, ov::Tensor& b, size_t blend_extent);

/**
 * @brief Blend two 5D tensors along height dimension (vertical seam removal)
 *
 * For video tensors with layout:
 * - [B, T, H, W, C] (BTHWC, u8) - postprocessed output
 * - [B, C, T, H, W] (BCTHW, f32) - raw output
 *
 * Blends the bottom blend_extent rows of tensor 'a' with top blend_extent rows of tensor 'b'
 * Result is written to tensor 'b'
 *
 * @param a First tensor (above)
 * @param b Second tensor (below) - modified in place
 * @param blend_extent Number of rows to blend
 */
void blend_v_5d(ov::Tensor& a, ov::Tensor& b, size_t blend_extent);

/**
 * @brief Blend two 5D tensors along width dimension (horizontal seam removal)
 *
 * For video tensors with layout:
 * - [B, T, H, W, C] (BTHWC, u8) - postprocessed output
 * - [B, C, T, H, W] (BCTHW, f32) - raw output
 *
 * Blends the rightmost blend_extent columns of tensor 'a' with leftmost blend_extent columns of tensor 'b'
 * Result is written to tensor 'b'
 *
 * @param a First tensor (left)
 * @param b Second tensor (right) - modified in place
 * @param blend_extent Number of columns to blend
 */
void blend_h_5d(ov::Tensor& a, ov::Tensor& b, size_t blend_extent);

/**
 * @brief Auto-detect tensor dimensions and blend along height dimension
 *
 * Automatically dispatches to blend_v_4d or blend_v_5d based on tensor rank
 *
 * @param a First tensor (above)
 * @param b Second tensor (below) - modified in place
 * @param blend_extent Number of rows to blend
 */
void blend_v(ov::Tensor& a, ov::Tensor& b, size_t blend_extent);

/**
 * @brief Auto-detect tensor dimensions and blend along width dimension
 *
 * Automatically dispatches to blend_h_4d or blend_h_5d based on tensor rank
 *
 * @param a First tensor (left)
 * @param b Second tensor (right) - modified in place
 * @param blend_extent Number of columns to blend
 */
void blend_h(ov::Tensor& a, ov::Tensor& b, size_t blend_extent);

}  // namespace blend_utils
}  // namespace module
}  // namespace genai
}  // namespace ov
