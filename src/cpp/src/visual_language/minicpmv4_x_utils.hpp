// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/vision_encoder.hpp"

namespace ov::genai::minicpmv4_x_utils {

int ensure_divide(double length, int divisor);

// Keeps fractional resized dimensions until rounding to patch_size * merge_factor multiples.
ImageSize find_best_resize(
    double height,
    double width,
    int scale_resolution,
    int patch_size,
    int merge_factor,
    bool allow_upscale);

// Aligns the source to the slices grid and sizes each slice using find_best_resize.
ImageSize get_refine_size(
    int height,
    int width,
    const SlicesGrid& grid,
    int scale_resolution,
    int patch_size,
    int merge_factor);

// Selects an aspect-ratio-matched grid, preferring more rows on ties; empty means no slicing.
SlicesGrid get_slices_grid(int height, int width, int max_slice_nums, int scale_resolution);

} // namespace ov::genai::minicpmv4_x_utils
